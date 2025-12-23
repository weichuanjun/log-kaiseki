import os
from typing import Annotated, Literal, TypedDict

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
    # メッセージ履歴以外に、各ステップの完了状態などを持ちたい場合はここに追加
    pass

# ==========================================
# 3. ノード (処理ステップ) の定義
# ==========================================

def context_node(state: LogAnalysisState):
    """
    ステップ1: システム情報のコンテキストを注入するノード
    """
    messages = state["messages"]
    
    # システム情報をSystemMessageとして先頭に追加（または更新）
    # 既にSystemMessageがある場合は、それを維持しつつ、システム固有情報を追加する形にする
    # ここではシンプルに、CONTEXT_AGENT_PROMPT を含む SystemMessage を生成します。
    
    system_info = SystemMessage(content=CONTEXT_AGENT_PROMPT)
    
    # 履歴の先頭にシステムメッセージがない、または内容が異なれば追加
    # 注: LangGraphのstate reducerがappendなので、単純に返すと追加される。
    # 履歴の最初に挿入したいが、MessagesStateは追記型。
    # ここでは「コンテキストエージェントからの情報提供」としてメッセージを追加します。
    
    return {"messages": [system_info]}

def analysis_node(state: LogAnalysisState):
    """
    ステップ2: エラー解析を行うノード
    """
    messages = state["messages"]
    
    # 解析エージェントへの指示を追加
    instruction = HumanMessage(content=ANALYSIS_AGENT_PROMPT)
    
    # LLMを実行 (これまでの履歴 + 今回の指示)
    response = model.invoke(messages + [instruction])
    
    # AIの応答（解析結果）を返す
    return {"messages": [response]}

def summary_node(state: LogAnalysisState):
    """
    ステップ3: 最終レポートを作成するノード
    """
    messages = state["messages"]
    
    # 要約エージェントへの指示を追加
    instruction = HumanMessage(content=SUMMARY_AGENT_PROMPT)
    
    # LLMを実行 (これまでの履歴 + 今回の指示)
    response = model.invoke(messages + [instruction])
    
    # AIの応答（レポート）を返す
    return {"messages": [response]}

# ==========================================
# 4. グラフ (ワークフロー) の構築
# ==========================================

# グラフのビルダーを初期化
workflow = StateGraph(LogAnalysisState)

# ノードを追加
workflow.add_node("context_agent", context_node)
workflow.add_node("analysis_agent", analysis_node)
workflow.add_node("summary_agent", summary_node)

# エッジを定義 (リニアな流れ)
# START -> context -> analysis -> summary -> END
workflow.add_edge(START, "context_agent")
workflow.add_edge("context_agent", "analysis_agent")
workflow.add_edge("analysis_agent", "summary_agent")
workflow.add_edge("summary_agent", END)

# メモリ（チェックポイント）の設定
memory = MemorySaver()

# コンパイル
app = workflow.compile(checkpointer=memory)
