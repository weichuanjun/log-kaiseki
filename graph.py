import os
from typing import Annotated, Literal, TypedDict
import operator

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# プロンプト定義をインポート
from prompts import (
    SYSTEM_CONTEXT_INFO,
    CONTEXT_AGENT_PROMPT,
    ANALYSIS_AGENT_PROMPT,
    ANALYSIS_FOLLOWUP_PROMPT,
    CRITIC_AGENT_PROMPT,
    SUMMARY_AGENT_PROMPT
)

# 環境変数の読み込み
from dotenv import load_dotenv
load_dotenv()

# ==========================================
# 1. LLMの初期化設定
# ==========================================
def get_llm():
    """
    環境変数に基づいて適切なLLMインスタンスを返します。
    """
    llm_type = os.getenv("LLM_TYPE", "openai")

    if llm_type == "azure":
        # Azure OpenAIの設定
        return AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            temperature=0
        )
    else:
        # 本家OpenAIの設定
        return ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0
        )

# LLMインスタンスを作成
model = get_llm()

# ==========================================
# 2. State (状態) の定義
# ==========================================
class LogAnalysisState(MessagesState):
    # レビューの反復回数を追跡
    revision_count: int

# ==========================================
# 3. ノード (処理ステップ) の定義
# ==========================================

from langchain_core.runnables import RunnableConfig

def context_node(state: LogAnalysisState):
    """
    ステップ1: システム情報のコンテキストを注入するノード
    """
    messages = state["messages"]
    
    # 既にシステム情報が含まれている場合は重複して追加しない
    # (ただし、revision_countのリセットは行う)
    has_system_info = any(
        isinstance(m, SystemMessage) and "システム概要" in str(m.content) 
        for m in messages
    )
    
    result = {"revision_count": 0}
    
    if not has_system_info:
        system_info = SystemMessage(content=CONTEXT_AGENT_PROMPT)
        result["messages"] = [system_info]
        
    return result

def analysis_node(state: LogAnalysisState, config: RunnableConfig):
    """
    ステップ2: エラー解析を行うノード
    """
    messages = state["messages"]
    
    # プロンプトの選択ロジック
    # 直前のメッセージがユーザーからのもの（かつ、履歴に既にAIの応答がある）場合はフォローアップとみなす
    last_message = messages[-1]
    has_ai_history = any(isinstance(m, AIMessage) for m in messages)
    
    if isinstance(last_message, HumanMessage) and has_ai_history:
        # ユーザーからのフォローアップ入力
        prompt_content = ANALYSIS_FOLLOWUP_PROMPT
    else:
        # 初回分析、またはCritiqueからのフィードバックループ
        prompt_content = ANALYSIS_AGENT_PROMPT
    
    # 解析エージェントへの指示を追加
    instruction = HumanMessage(content=prompt_content)
    
    # LLMを実行 (これまでの履歴 + 今回の指示)
    response = model.invoke(messages + [instruction], config=config)
    
    # AIの応答（解析結果）を返す
    return {"messages": [response]}

def critique_node(state: LogAnalysisState, config: RunnableConfig):
    """
    ステップ2.5: 分析結果を批評するノード（新規追加）
    """
    messages = state["messages"]
    
    # 批評エージェントへの指示
    instruction = HumanMessage(content=CRITIC_AGENT_PROMPT)
    
    # LLMを実行
    response = model.invoke(messages + [instruction], config=config)
    
    # 反復回数をインクリメント
    current_count = state.get("revision_count", 0)
    
    return {"messages": [response], "revision_count": current_count + 1}

def summary_node(state: LogAnalysisState, config: RunnableConfig):
    """
    ステップ3: 最終レポートを作成するノード
    """
    messages = state["messages"]
    
    # 要約エージェントへの指示を追加
    instruction = HumanMessage(content=SUMMARY_AGENT_PROMPT)
    
    # LLMを実行 (これまでの履歴 + 今回の指示)
    response = model.invoke(messages + [instruction], config=config)
    
    # AIの応答（レポート）を返す
    return {"messages": [response]}

# ==========================================
# 4. 条件付きエッジのロジック
# ==========================================

def should_continue(state: LogAnalysisState) -> Literal["analysis_agent", "summary_agent"]:
    """
    Critiqueの結果を見て、修正が必要か（analysisに戻るか）、完了か（summaryに進むか）を判断する
    """
    messages = state["messages"]
    last_message = messages[-1]
    content = last_message.content
    
    # 反復回数の上限チェック (無限ループ防止、最大1回までレビュー)
    # ユーザー要望: Critique Agentは一度だけ指摘する
    if state["revision_count"] >= 1:
        return "summary_agent"
    
    # 批評家が承認したかどうか
    if "APPROVE" in content:
        return "summary_agent"
    else:
        # REJECTの場合は再分析へ
        # 批評家のコメントが履歴に残っている状態でAnalysisに戻るので、LLMはそれを考慮して修正できる
        return "analysis_agent"

# ==========================================
# 5. グラフ (ワークフロー) の構築
# ==========================================

# グラフのビルダーを初期化
workflow = StateGraph(LogAnalysisState)

# ノードを追加
workflow.add_node("context_agent", context_node)
workflow.add_node("analysis_agent", analysis_node)
workflow.add_node("critique_node", critique_node)
workflow.add_node("summary_agent", summary_node)

# エッジを定義
# START -> context -> analysis -> critique
workflow.add_edge(START, "context_agent")
workflow.add_edge("context_agent", "analysis_agent")
workflow.add_edge("analysis_agent", "critique_node")

# 条件付きエッジ: critique -> (check) -> analysis OR summary
workflow.add_conditional_edges(
    "critique_node",
    should_continue,
)

workflow.add_edge("summary_agent", END)

# メモリ（チェックポイント）の設定
memory = MemorySaver()

# コンパイル
app = workflow.compile(checkpointer=memory)
