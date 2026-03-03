# Help Desk Voice Bot

A highly modular, decoupled Voice Bot leveraging Retrieval-Augmented Generation (RAG). The system features a modern FastAPI backend and a glassmorphic Streamlit frontend, allowing voice-to-voice interaction using state-of-the-art open models.

## Key Features

- **Decoupled Architecture**: Independent frontend and backend for easy scalability.
- **Provider Pattern**: Easy to swap out different LLMs, Vector Stores, and Voice models.
- **RAG Enabled**: Powered by LangChain and Pinecone.
- **Voice Capabilities**: Built-in support for Speech-To-Text (Whisper / OpenRouter) and Text-To-Speech (Edge-TTS).

## Project Structure

```text
Voice-bot/
├── backend/                  # FastAPI Application
│   ├── api/                  # Routes and endpoints
│   ├── core/                 # Config and base interfaces
│   ├── providers/            # Provider implementations and routing files
│   │   ├── vector_store.py   # Factory routing for Vector DBs (e.g. Pinecone)
│   │   ├── llm_provider.py   # Factory routing for LLMs (e.g. OpenRouter)
│   │   └── ... 
│   ├── rag/                  # RAG logic (Document loaders & LCEL Chains)
│   ├── voice/                # Audio processing orchestration
│   └── main.py               # Entrypoint
└── frontend/                 # Streamlit UI
    └── app.py                # UI Code
```

## Getting Started

### 1. Backend
Navigate to the `backend` folder:
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
```
Fill in the `.env` file with your **OpenRouter**, **Pinecone**, and (optional) **HuggingFace** keys. Then start the server:
```bash
uvicorn main:app --reload
```

### 2. Frontend
In a new terminal, navigate to the `frontend` folder:
```bash
cd frontend
pip install -r requirements.txt
cp .env.example .env
```
Start the Streamlit application:
```bash
streamlit run app.py
```

## Modularity & Providers
The project uses routing files like `backend/providers/vector_store.py` and `backend/providers/llm_provider.py` to dynamically load the implementations specified in your `.env`. The API uses **dependency injection** (`backend/api/dependencies.py`) so providers are created once and reused.

If you wish to change from `openrouter` STT to `local_whisper`, simply update the `STT_PROVIDER=local_whisper` setting in your `.env` without modifying any code!

## Testing
From the **Voice-bot** directory (project root), with backend dependencies installed:

```bash
cd Voice-bot
pip install -r backend/requirements.txt -r backend/requirements-dev.txt
PYTHONPATH=. python -m pytest tests/ -v
```

Some tests are skipped when optional packages (e.g. `edge_tts`, `langchain_huggingface`, `whisper`) are not installed. For full API tests, ensure the backend `.env` is configured and all dependencies are installed.
