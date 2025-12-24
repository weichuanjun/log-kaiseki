import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_core.runnables.config import RunnableConfig

# 定義したLangGraphアプリケーションをインポート
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

@cl.on_chat_start
async def on_chat_start():
    """
    チャットセッション開始時の処理。
    ユーザーにファイルを要求するダイアログを表示します。
    """
    files = None
    
    # ファイルがアップロードされるまで待機するループ（必要であれば）
    # ここでは一度だけ尋ねる形式にします
    while files is None:
        files = await cl.AskFileMessage(
            content="分析したいログファイルをアップロードしてください（複数選択可）。\nフォルダ内の全ファイルを分析する場合は、それらすべてを選択してアップロードしてください。",
            accept={"*/*": []},
            max_size_mb=100,
            max_files=1000,
            timeout=600
        ).send()

    # ファイルがアップロードされたら処理を開始
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
                    # 1-based index for user friendliness
                    content_with_lines += f"{i+1:04d}: {line}"
                
                log_contents += f"\n\n=== ログファイル: {file.name} ===\n{content_with_lines}\n==========================\n"
        except UnicodeDecodeError:
             log_contents += f"\n\n=== ログファイル: {file.name} ===\n(エンコードエラー: UTF-8テキストとして読み込めませんでした)\n==========================\n"
        except Exception as e:
            log_contents += f"\n\n=== ログファイル: {file.name} ===\n(読み込みエラー: {str(e)})\n==========================\n"
            
    # ユーザーに読み込み完了を通知
    await cl.Message(
        content=f"以下の{len(file_names)}個のファイルを読み込みました:\n{', '.join(file_names)}\n\n分析を開始します..."
    ).send()

    # LangGraphを実行して初期分析を行う
    user_input_content = f"以下のログファイルを分析してください:\n{', '.join(file_names)}\n{log_contents}"
    
    answer = cl.Message(content="")
    await answer.send()

    config: RunnableConfig = {
        "configurable": {"thread_id": cl.context.session.thread_id}
    }

    async for msg, _ in app_graph.astream(
        {"messages": [HumanMessage(content=user_input_content)]},
        config,
        stream_mode="messages",
    ):
        if isinstance(msg, AIMessageChunk):
            answer.content += msg.content
            await answer.update()


@cl.on_chat_resume
async def on_chat_resume(thread):
    """
    過去のセッション再開時の処理。
    """
    pass

@cl.on_message
async def main(message: cl.Message):
    """
    ユーザーからのメッセージ受信時のメイン処理。
    追加の質問やファイルアップロードを処理します。
    """
    
    # 応答用の空メッセージを作成
    answer = cl.Message(content="")
    
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

    # 3. LangGraphの実行とストリーミング
    await answer.send()

    config: RunnableConfig = {
        "configurable": {"thread_id": cl.context.session.thread_id}
    }

    async for msg, _ in app_graph.astream(
        {"messages": [HumanMessage(content=user_input_content)]},
        config,
        stream_mode="messages",
    ):
        if isinstance(msg, AIMessageChunk):
            answer.content += msg.content
            await answer.update()
