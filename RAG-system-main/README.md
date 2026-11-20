# FinRAG AI

FinRAG AI is a local financial-document question-answering application. It extracts text from uploaded files, indexes it in ChromaDB, retrieves evidence for a question, and asks a local Ollama model to answer from that evidence.

## Features

- User registration and login with bearer-token authentication.
- Per-user document indexing and retrieval.
- Upload TXT, PDF, DOCX, image, and audio files (up to 20 MB each).
- PDF/DOCX/TXT text extraction, Tesseract OCR for images, and Whisper transcription for audio.
- ChromaDB vector search with company-aware and cross-company retrieval.
- Follow-up questions using the active chat session's company context.
- Local answer generation with Ollama (`llama3`).
- Frontend display of the number of indexed files for the signed-in user.

Images and audio are searched as extracted text, not as raw pixels or sound. Answer quality depends on the quality of the source file and its extracted text.

## Technology

- Python, FastAPI, LangChain, and ChromaDB
- Hugging Face `sentence-transformers/all-MiniLM-L6-v2` embeddings
- Ollama with the `llama3` model
- TypeScript, Vite, and npm
- SQLite for user accounts

## Requirements

- Python and Node.js/npm installed.
- Python dependencies can be installed from the repository root with `python -m pip install -r requirements.txt`.
- Ollama installed, running locally, and configured with the `llama3` model.
- The embedding model available in the local Hugging Face cache; retrieval is configured to load it locally.
- Tesseract OCR installed for image uploads.
- FFmpeg available on `PATH` for audio transcription. The upload processor also checks `C:\ffmpeg-8.1.1-essentials_build\bin`.
- A root `.env` file containing the required application secrets.

## Configuration

Create `.env` in the repository root. Do not commit it or share its values.

```dotenv
FINRAG_SECRET_KEY=<random secret of at least 40 characters>
FINRAG_ADMIN_USERNAME=<initial admin username>
FINRAG_ADMIN_PASSWORD=<initial admin password>
FINRAG_ADMIN_EMAIL=<initial admin email>
```

The application seeds the admin account when the user database is initialized and that admin account does not already exist. Changing these variables later does not change an existing account's password. The user database is stored at `db/users.sqlite3`.

ChromaDB data is stored in `db/chroma_db`. Uploaded documents are associated with the authenticated user's ID; do not delete or replace this directory if you need to preserve indexed documents.

## Run locally

Start Ollama and make sure `llama3` is available. Then open two terminals at the repository root.

Terminal 1 — backend:

```powershell
.\venv\Scripts\Activate.ps1
uvicorn backend.app:app --reload
```

Terminal 2 — frontend:

```powershell
Set-Location frontend
npm install
npm run dev
```

Open the local URL printed by Vite (normally `http://localhost:5173`). The frontend is configured to call the API at `http://127.0.0.1:8000`; the backend CORS configuration allows the Vite localhost origins.

## Main API routes

- `GET /` and `GET /health` — service status.
- `POST /register` and `POST /login` — account registration and authentication.
- `POST /upload` — upload and index one file (authenticated).
- `GET /uploads/count` — count distinct indexed source files for the signed-in user (authenticated).
- `POST /ask` — retrieve evidence and generate an answer (authenticated).

Interactive API documentation is available at `http://127.0.0.1:8000/docs` while the backend is running.

## Notes

- A retrieved-chunk count is not an accuracy score. Answers can be incomplete when the source has extraction errors or retrieval misses relevant evidence.
- The backend currently uses in-memory conversation context, so follow-up context is reset when the backend restarts.
- This project is configured for local development. Review authentication, CORS, secrets, file handling, and deployment settings before exposing it beyond your machine.
