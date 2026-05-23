# Adaptive RAG AI Learning Platform

> A full-stack AI learning platform that combines **DistilBERT intent classification**, **LangGraph adaptive routing**, **local LLM inference with Ollama**, and a **modern Angular frontend** to dynamically choose the best retrieval strategy for each user query.

---

## Preview

```text
Debugging Query  → Web Search Route
Theory Question  → Vectorstore Route
```

The system visually exposes:

* predicted intent
* confidence score
* selected graph route
* retrieval strategy
* classifier traces

inside the frontend UI in real time.

---

# Features

## Adaptive Query Routing

Queries are dynamically routed using a fine-tuned DistilBERT classifier.

| Query Type              | Route                 |
| ----------------------- | --------------------- |
| Debugging / coding help | Web Search            |
| Theory / concepts       | Vectorstore Retrieval |

---

## Full Stack Architecture

### Frontend

* Angular 19
* TypeScript
* JWT auth
* Responsive cyberpunk UI
* Live routing visualization

### Backend

* FastAPI
* LangGraph orchestration
* ChromaDB vector retrieval
* SQLite authentication
* Ollama local LLM inference

### ML / AI

* DistilBERT intent classifier
* Transformers + PyTorch
* Adaptive RAG pipeline

---

# System Architecture

```text
Angular Frontend (4200)
        ↓
Adaptive RAG Backend (8001)
        ↓
Intent Classifier Service (8000)
        ↓
LangGraph Routing Engine
        ↓
├── Vectorstore Retrieval (ChromaDB)
└── Web Search Route
        ↓
Ollama Local LLM
```

---

# Key Capabilities

* Intent-aware adaptive retrieval
* JWT-secured authentication
* LangGraph multi-stage orchestration
* Live route transparency
* Query history
* Responsive UI
* Local offline inference
* Real-time classifier confidence display
* Full-stack AI workflow visualization

---

# Demo Flow

## Theory Question

Input:

```text
What is machine learning?
```

Expected Route:

```text
vectorstore retrieval
```

Displayed Metadata:

* predicted_intent
* confidence score
* retrieval strategy
* graph route

---

## Debugging Question

Input:

```text
How do I fix this Python traceback error?
```

Expected Route:

```text
websearch
```

Displayed Metadata:

* predicted_intent
* confidence score
* retrieval strategy
* graph route

---

# Project Structure

```text
Project 3/
│
├── frontend/                         # Angular frontend
│
├── edu-intent-classifier/            # DistilBERT classifier API
│
├── Adaptive_RAG-Fastapi-Langgraph/  # LangGraph RAG backend
│
└── README.md
```

---

# Running the Project

## Prerequisites

* Python 3.11
* Node.js + npm
* Ollama installed locally
* Ports available:

  * 4200
  * 8000
  * 8001

---

# Start Backend Services


### Terminal 1 — Intent Classifier
```powershell
cd "c:/Users/Sree/Desktop/Internship/Essentials/Projects/Project 3/edu-intent-classifier"
& "../.venv/Scripts/python.exe" -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### Terminal 2 — Adaptive RAG Backend
```powershell
cd "c:/Users/Sree/Desktop/Internship/Essentials/Projects/Project 3/Adaptive_RAG-Fastapi-Langgraph"
& ".venv\Scripts\python.exe" run_server.py
```

### Terminal 3 — Angular Frontend
```powershell
cd "c:/Users/Sree/Desktop/Internship/Essentials/Projects/Project 3/frontend"
npx @angular/cli serve
```

---

# Application URLs

| Service                 | URL                                                              |
| ----------------------- | ---------------------------------------------------------------- |
| Angular Frontend        | [http://localhost:4200](http://localhost:4200)                   |
| Login Page              | [http://localhost:4200/login](http://localhost:4200/login)       |
| Register Page           | [http://localhost:4200/register](http://localhost:4200/register) |
| RAG Swagger Docs        | [http://localhost:8001/docs](http://localhost:8001/docs)         |
| Classifier Swagger Docs | [http://localhost:8000/docs](http://localhost:8000/docs)         |

---

# API Endpoints

## Authentication

```text
POST /token
POST /register
GET  /users/me
```

## Intent Classifier

```text
POST /predict
GET  /health
```

## Adaptive RAG

```text
POST /query
GET  /health
```

---

# LangGraph Workflow

The backend uses a multi-stage LangGraph pipeline:

```text
Route → Retrieve → Generate → Grade
```

### Route Stage

Determines the optimal retrieval path using classifier output.

### Retrieve Stage

Fetches relevant context from:

* ChromaDB
* web search

### Generate Stage

Uses Ollama to generate the final answer.

### Grade Stage

Evaluates response quality and retries if needed.

---

# Authentication Flow

```text
Frontend Login
    ↓
POST /token
    ↓
JWT Generated
    ↓
Stored in Frontend
    ↓
Bearer Token attached to /query requests
```

---

# Why This Project Is Unique

* Custom fine-tuned DistilBERT classifier
* Adaptive ML-driven routing
* LangGraph orchestration pipeline
* Local LLM execution using Ollama
* Full-stack implementation
* Transparent AI routing visualization
* No dependency on paid cloud AI APIs

---

# Future Improvements

* Docker deployment
* Redis caching
* Streaming token responses
* Multi-user support
* Voice interaction
* Cloud vector database
* CI/CD pipeline
* Kubernetes deployment

---

# Author

## Hemanth Harish

GitHub:
[https://github.com/HemanthHarish07/adaptive-rag-ai-platform](https://github.com/HemanthHarish07/adaptive-rag-ai-platform)
