# Log Kaiseki Agent (ログ解析エージェント)

LangGraphとChainlitを使用した、高度なログ解析システムです。
複数のAIエージェント（Context, Analysis, Critique, Summary）が協力して、アップロードされたログファイルを分析し、エラーの原因と解決策を特定します。

## 機能

- **ログ一括分析**: フォルダごとのログファイルを一括でアップロードし、分析できます。
- **マルチエージェント協調**: 
  - **Context Agent**: システム情報の確認
  - **Analysis Agent**: エラーの詳細分析（ファイル名・行番号の引用付き）
  - **Critique Agent**: 分析結果のレビューと改善指摘（自己修正ループ）
  - **Summary Agent**: 最終的な専門家レポートの作成
- **可視化された思考プロセス**: 各エージェントの処理内容がリアルタイムでUIに表示されます。
- **マルチLLM対応**: OpenAI (GPT-4o) および Azure OpenAI に対応。拡張も容易です。

## セットアップ手順

### 1. 前提条件

- Python 3.10 以上
- OpenAI API Key または Azure OpenAI 環境

### 2. 仮想環境の作成と有効化

プロジェクトのルートディレクトリで以下のコマンドを実行します。

```bash
# 仮想環境の作成
python -m venv venv

# 有効化 (Mac/Linux)
source venv/bin/activate

# 有効化 (Windows)
.\venv\Scripts\activate
```

### 3. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 4. 環境変数の設定

`example.env` をコピーして `.env` を作成し、必要な情報を入力します。

```bash
cp example.env .env
```

`.env` ファイルの内容:

```env
# LLMの選択 (openai または azure)
LLM_TYPE=openai

# OpenAIを使用する場合
OPENAI_API_KEY=sk-proj-...

# Azure OpenAIを使用する場合
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

# Chainlitの認証シークレット (生成コマンド: chainlit create-secret)
CHAINLIT_AUTH_SECRET=...
```

### 5. アプリケーションの起動

```bash
chainlit run app.py -w
```

ブラウザが自動的に開き、チャット画面が表示されます。

## 新しいLLMノードの追加方法（例：Google Gemini）

新しいLLM（例：Google Gemini）を追加したい場合は、以下の手順でコードを修正してください。

1.  **依存関係の追加**:
    `requirements.txt` に `langchain-google-genai` を追加してインストールします。

2.  **`graph.py` の修正**:
    `get_llm` 関数に分岐を追加します。

    ```python
    from langchain_google_genai import ChatGoogleGenerativeAI

    def get_llm():
        llm_type = os.getenv("LLM_TYPE", "openai")

        if llm_type == "google":
            return ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0
            )
        # ... 既存のコード ...
    ```

3.  **`.env` の更新**:
    `LLM_TYPE=google` と `GOOGLE_API_KEY` を設定します。

## プロジェクト構成

- `app.py`: Chainlit UIとアプリケーションのエントリーポイント
- `graph.py`: LangGraphによるエージェントワークフローの定義
- `prompts.py`: 各エージェントのプロンプト定義
- `requirements.txt`: 依存ライブラリ一覧
