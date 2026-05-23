# Adaptive RAG AI Learning Platform

Adaptive RAG learning assistant combining a local intent classifier (8000), a LangGraph RAG backend (8001), and an Angular UI (4200) with browser-exposed routing metadata.

## Architecture (text-based diagram)

Browser (4200) → Angular Frontend

Angular → RAG Backend (8001) → Classifier (8000)
                                 → Ollama (local LLM)
                                 → ChromaDB (vectors)

## Tech Stack

- **Angular** (frontend, port `4200`)
- **FastAPI + LangGraph** (RAG backend, port `8001`)
- **FastAPI (intent classifier service)** on port `8000`
- **Ollama** (local LLM)
- **ChromaDB** (vector store)
- **SQLite** (JWT user store)
- **JWT auth** (token-based security)

## Prerequisites

- Python 3.11+
- Node.js + npm (for Angular)
- Local **Ollama** running with at least one model available
- Ports free: **8000**, **8001**, **4200**

## Quick Start (3 terminals)

Open 3 separate PowerShell terminals from the repo root.

### Terminal 1 — classifier on 8000

```powershell
Set-Location "c:/Users/Sree/Desktop/Internship/Essentials/Projects/Project 3/Adaptive_RAG-Fastapi-Langgraph"
& ".venv\Scripts\python.exe" "..\edu-intent-classifier\run.py" --serve
```

### Terminal 2 — RAG backend on 8001

```powershell
Set-Location "c:/Users/Sree/Desktop/Internship/Essentials/Projects/Project 3/Adaptive_RAG-Fastapi-Langgraph"
& ".venv\Scripts\python.exe" "run_server.py"
```

### Terminal 3 — ng serve (frontend on 4200)

```powershell
Set-Location "c:/Users/Sree/Desktop/Internship/Essentials/Projects/Project 3/frontend"
ng serve
```

## Browser URLs

- Frontend app:      http://localhost:4200
- Login page:        http://localhost:4200/login
- Register page:     http://localhost:4200/register
- RAG Swagger docs:  http://localhost:8001/docs
- Classifier docs:   http://localhost:8000/docs
- RAG health:        http://localhost:8001/health
- Classifier health: http://localhost:8000/health

## How to Use the App (card)

### Test credentials

- **Username:** testuser
- **Password:** test123

### 1) Login

1. Open: http://localhost:4200/login
2. Enter the credentials above.
3. Submit.

### 2) Run queries & see routing metadata

After login, use the query box in the main UI.

#### A) Route that goes to **vectorstore**

- **Type in the query box:**
  - `What is machine learning?`
- **What routing metadata panel should show:**
  - `predicted_intent`: something concept/definition-like (e.g., “concept_definition” / similar)
  - `intent_confidence`: a non-null confidence score (e.g., 0.5+)
  - `selected_graph_route`: `vectorstore` (or the UI’s label for vector retrieval)
  - `retrieval_strategy`: `vectorstore` / `chroma` (depending on the UI wording)
  - `classifier_request_trace`: the classifier trace returned by the classifier service (non-empty)

#### B) Route that goes to **websearch**

- **Type in the query box:**
  - `How do I fix this error in Python when I get a traceback?`
- **What routing metadata panel should show:**
  - `predicted_intent`: something debugging/help-like (e.g., “debug_help” / similar)
  - `intent_confidence`: a non-null confidence score
  - `selected_graph_route`: `websearch`
  - `retrieval_strategy`: `websearch` / external web retrieval mode (depending on UI wording)
  - `classifier_request_trace`: non-empty

### Notes on routing metadata

The UI reads routing metadata that the backend returns in the query response (e.g., `predicted_intent`, `selected_graph_route`, `retrieval_strategy`, and `classifier_request_trace`).

## API Endpoints

### Authentication (RAG backend - `8001`)

- `POST /token` — login and receive JWT
- `GET /users/me` — current authenticated user info

### Intent Classifier service (`8000`)

- `POST /predict` — predict intent for a query
- `GET /health` — health check

### RAG System (`8001`)

- `POST /query` — run retrieval-augmented generation
- `GET /health` — health check

## Features

- **FastAPI** server with `/query`, `/token`, and health endpoints
- **LangGraph** multi-stage orchestration for routing, retrieval, generation, and grading
- **Local Ollama** LLM integration
- **SQLite** user store for JWT authentication
- **Intent classification** microservice on port `8000`
- **Adaptive routing** from classifier output to either **vectorstore** retrieval or **websearch**
- **JWT auth flow** enforced by the RAG backend
- UI displays **routing metadata** for transparency and debugging

## LangGraph pipeline explanation

The RAG backend runs a LangGraph state machine across stages:

1. **Route stage**: chooses the best route based on classifier output.
2. **Retrieve stage**: fetches context from **ChromaDB** (vectorstore) or prepares web context.
3. **Generate stage**: generates the final answer using **Ollama**.
4. **Grade stage**: evaluates/refines the answer quality.

The backend streams events internally and then returns final `answer` plus routing metadata fields.

## JWT auth flow

1. User logs in via the frontend -> `POST http://localhost:8001/token`.
2. Backend verifies username/password against the **SQLite** user store.
3. Backend returns a signed **JWT**.
4. Frontend stores the token (local storage) and sends it as:
   - `Authorization: Bearer <JWT>`
5. Backend protects `POST /query` using `OAuth2PasswordBearer`.

## Routing behavior explanation

- The backend first calls the classifier service (`8000`) to get intent and routing decision.
- Then the LangGraph route selection determines whether the query will use:
  - **Vectorstore** retrieval for concept/definition questions
  - **Websearch** for debugging/coding-help-like questions
- The response includes routing metadata so the UI can show:
  - `predicted_intent`
  - `selected_graph_route`
  - `retrieval_strategy`
  - `classifier_request_trace`

## Notes

- Designed for local development and experimentation.
- Backend uses **SQLite** (no PostgreSQL needed).
- If the RAG UI doesn’t show routing metadata, ensure backend and classifier are running and reachable on `8001`/`8000`.
- Make sure Ollama has a working model selected/available for generation.

## Why This Project Is Unique

- Only project in this portfolio with a custom fine-tuned model (DistilBERT)
- Fully offline: no cloud API required, runs on local hardware
- LangGraph orchestration: not a simple API call, a real stateful pipeline
- Adaptive routing: queries take different paths based on ML classification
- Full stack: trained model → FastAPI microservice → Angular UI


