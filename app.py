import chainlit as cl
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from graph import app as app_graph

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """
    ユーザー認証コールバック。
    現在は単純な admin/admin チェックのみ。
    """
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None

async def run_analysis_graph(user_input_content: str, config: RunnableConfig):
    """
    LangGraphを実行し、思考プロセスと最終結果をChainlitにストリーミングする共通関数。
    """
    
    final_answer = None
    last_node = None
    current_step = None
    
    # ノード名と表示名のマッピング
    node_map = {
        "context_agent": "Context Agent",
        "analysis_agent": "Analysis Agent",
        "critique_node": "Critique Agent",
        "summary_agent": "Summary Agent"
    }

    # イベントストリームの処理
    async for event in app_graph.astream_events(
        {"messages": [HumanMessage(content=user_input_content)]},
        config,
        version="v2"
    ):
        kind = event["event"]
        node_name = event.get("metadata", {}).get("langgraph_node")
        
        if not node_name:
            continue
            
        # --- ノード開始 (Agent Start) ---
        if kind == "on_chain_start" and node_name != last_node:
            # 前のステップがあれば閉じる
            if current_step:
                await current_step.update()
                current_step = None

            # Summary Agentに到達したら、最終回答の準備をする
            if node_name == "summary_agent":
                final_answer = cl.Message(content="")
                await final_answer.send()
            else:
                # 新しいエージェントステップを開始 (type="process"で折りたたみ可能)
                display_name = node_map.get(node_name, node_name)
                current_step = cl.Step(name=display_name, type="process")
                await current_step.send()
                
                # ログ風の開始ヘッダー
                header = f"```text\n=== [START] {display_name} ===\n"
                await current_step.stream_token(header)
            
            last_node = node_name
            
        # --- ストリーミング (Live Streaming) ---
        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            
            if not content:
                continue
            
            if node_name == "summary_agent":
                # 最終回答はそのままストリーミング
                if final_answer:
                    await final_answer.stream_token(content)
            else:
                # 思考プロセス中は、内容をそのまま流すが、コードブロック内にあるのでログとして見える
                if current_step:
                    await current_step.stream_token(content)

        # --- ノード終了 (Agent End) ---
        elif kind == "on_chain_end" and node_name in node_map and node_name != "summary_agent":
            # ステップの終了処理
            if current_step:
                footer = f"\n=== [END] {node_map.get(node_name, node_name)} ===\n```"
                await current_step.stream_token(footer)
                await current_step.update()
                current_step = None

    # ループ終了後の後処理
    if current_step:
        await current_step.update()

    if final_answer:
        # 最終回答メッセージを完了（UI上の更新）
        await final_answer.update()
    else:
        # 万が一Summary Agentが実行されずに終了した場合の安全策
        if not final_answer:
             final_answer = cl.Message(content="処理が完了しましたが、応答が生成されませんでした。")
             await final_answer.send()

@cl.on_chat_start
async def on_chat_start():
    """
    チャットセッション開始時の処理。
    """
    files = None
    
    # ファイルアップロード待機
    while files is None:
        files = await cl.AskFileMessage(
            content="分析したいログファイルをアップロードしてください（複数選択可）。\nフォルダ内の全ファイルを分析する場合は、それらすべてを選択してアップロードしてください。",
            accept={"*/*": []},
            max_size_mb=100,
            max_files=1000,
            timeout=600
        ).send()

    # ファイル読み込み処理
    log_contents = ""
    file_names = []
    
    for file in files:
        file_names.append(file.name)
        try:
            with open(file.path, "r", encoding="utf-8") as f:
                # 行番号を付与して読み込む
                lines = f.readlines()
                content_with_lines = ""
                for i, line in enumerate(lines):
                    # 1-based index
                    content_with_lines += f"{i+1:04d}: {line}"
                
                log_contents += f"\n\n=== ログファイル: {file.name} ===\n{content_with_lines}\n==========================\n"
        except UnicodeDecodeError:
             log_contents += f"\n\n=== ログファイル: {file.name} ===\n(エンコードエラー: UTF-8テキストとして読み込めませんでした)\n==========================\n"
        except Exception as e:
            log_contents += f"\n\n=== ログファイル: {file.name} ===\n(読み込みエラー: {str(e)})\n==========================\n"
            
    # 読み込み完了通知
    await cl.Message(
        content=f"以下の{len(file_names)}個のファイルを読み込みました:\n{', '.join(file_names)}\n\n分析を開始します..."
    ).send()

    # LangGraph入力の構築
    user_input_content = f"以下のログファイルを分析してください:\n{', '.join(file_names)}\n{log_contents}"
    
    config: RunnableConfig = {
        "configurable": {"thread_id": cl.context.session.thread_id}
    }
    
    # 共通関数で分析実行
    await run_analysis_graph(user_input_content, config)

@cl.on_message
async def main(message: cl.Message):
    """
    ユーザーからのメッセージ受信時のメイン処理。
    """
    
    # 1. 添付ファイルの処理 (追加でアップロードされた場合)
    log_contents = ""
    file_names = []
    
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File):
                file_names.append(element.name)
                try:
                    with open(element.path, "r", encoding="utf-8") as f:
                        content = f.read()
                        log_contents += f"\n\n=== Log File: {element.name} ===\n{content}\n==========================\n"
                except Exception as e:
                    log_contents += f"\n\n=== Log File: {element.name} ===\n(Read Error: {str(e)})\n==========================\n"

    # 2. ユーザー入力の構築
    user_input_content = message.content
    
    if log_contents:
        user_input_content += f"\n\n(追加ログファイル):\n{', '.join(file_names)}\n{log_contents}"
        await cl.Message(content=f"追加のファイルを読み込みました: {', '.join(file_names)}").send()

    # 3. LangGraphの実行
    config: RunnableConfig = {
        "configurable": {"thread_id": cl.context.session.thread_id}
    }
    
    # 共通関数で分析実行
    await run_analysis_graph(user_input_content, config)
