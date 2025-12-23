# ログ解析エージェント (Log-Kaiseki Agent)

ChainlitとLangGraphを使用した、インテリジェントなログ分析エージェントです。
複数のログファイルをアップロードすると、AIがエラーの特定、原因分析、解決策の提案を行い、マネージャー向けの要約レポートも生成します。

## 機能

*   **マルチログ分析**: 複数のログファイル（`access.log`, `app.log`, `db.log` など）を一度に分析。
*   **文脈理解**: システムの基本情報（`prompts.py`で定義）に基づいた専門的な分析。
*   **3段階のエージェントワークフロー**:
    1.  **Context Agent**: システム情報の注入
    2.  **Analysis Agent**: 技術的な詳細分析
    3.  **Summary Agent**: マネージャー向け要約レポート生成
*   **マルチLLM対応**: OpenAI (GPT-4o) と Azure OpenAI を設定で切り替え可能。

## セットアップガイド

### 1. 仮想環境の作成と有効化

```bash
# 仮想環境の作成
python3 -m venv venv

# 仮想環境の有効化 (Mac/Linux)
source venv/bin/activate

# 仮想環境の有効化 (Windows)
# venv\Scripts\activate
```

### 2. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 3. 環境変数の設定

`example.env` をコピーして `.env` ファイルを作成し、APIキーを設定してください。

```bash
cp example.env .env
```

`.env` の内容を編集します：

```env
# Chainlitの認証シークレット（必須）
# 生成コマンド: chainlit create-secret
CHAINLIT_AUTH_SECRET=your_generated_secret_here

# LLMの選択 (openai, azure, google)
LLM_TYPE=openai

# OpenAIを使用する場合
OPENAI_API_KEY=sk-your-openai-api-key

# Azure OpenAIを使用する場合
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

# Google Geminiを使用する場合 (拡張用)
GOOGLE_API_KEY=your_google_api_key
```

### 4. アプリケーションの起動

```bash
chainlit run app.py
```
ブラウザが自動的に開き、`http://localhost:8000` にアクセスできます。
デフォルトのログイン情報は `admin` / `admin` です。

## 拡張ガイド

### 新しいLLMノード（例：Google Gemini）を追加する方法

1.  **ライブラリのインストール**:
    Googleのモデルを使用するために必要なライブラリ（例: `langchain-google-genai`）をインストールし、`requirements.txt` に追加します。

2.  **`graph.py` の修正**:
    `get_llm()` 関数にGoogleモデルの分岐を追加します。

    ```python
    # graph.py

    # 必要なライブラリをインポート
    from langchain_google_genai import ChatGoogleGenerativeAI

    def get_llm():
        llm_type = os.getenv("LLM_TYPE", "openai")

        if llm_type == "google":
            return ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0
            )
        elif llm_type == "azure":
            # ... (既存のコード)
        else:
            # ... (既存のコード)
    ```

3.  **`.env` の更新**:
    `LLM_TYPE=google` と `GOOGLE_API_KEY` を設定します。

### システム情報のカスタマイズ

分析対象のシステムが変わった場合は、`prompts.py` の `SYSTEM_CONTEXT_INFO` 変数を書き換えてください。
ログの形式や、特有のエラーパターンなどを記述することで、AIの分析精度が向上します。
