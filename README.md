# Adaptive RAG AI Platform
Built as a full-stack AI learning platform demonstrating intent-aware retrieval, LangGraph orchestration, and local LLM inference using Ollama.
Intent-Aware Retrieval-Augmented Generation (RAG) system using DistilBERT, LangGraph, FastAPI, Angular, and Ollama.

## Verified Working State

This repository has been verified working end-to-end with the following verified components:

- **Frontend (Angular)** running on **http://localhost:4200**
- **Intent Classifier (FastAPI)** running on **http://localhost:8000**
  - Model loads successfully
  - **POST `/predict`** works
- **Adaptive RAG Backend (FastAPI)** running on **http://localhost:8001**
  - **GET `/health`** works
  - **POST `/query`** works
- **Ollama** running with the verified installed model:
  - `qwen2.5-coder:3b`

## Overview

Adaptive RAG AI Learning Platform is a full-stack AI learning platform that dynamically chooses retrieval strategies based on query intent and exposes routing transparency to the user.

It combines:

- **DistilBERT Intent Classification**
- **Adaptive RAG Routing**
- **LangGraph Workflow**
- **FastAPI Backend**
- **Ollama Local LLM Inference**
- **Angular Frontend**

## Features

- DistilBERT Intent Classification
- Intent-Aware Query Routing
- Adaptive Retrieval Routing
- Retrieval-Augmented Generation (RAG) Pipeline
- LangGraph Workflow Orchestration
- FastAPI Microservices
- JWT Authentication
- Ollama Local LLM Integration
- Angular Frontend
- Real-Time Routing Transparency

## Architecture

```text
Angular Frontend (4200)
        ↓
Adaptive RAG Backend (8001)
        ↓
Intent Classifier (8000)
        ↓
LangGraph Workflow
        ↓
Vector Retrieval / Search
        ↓
Ollama LLM
```

The classifier predicts user intent, the LangGraph workflow selects an appropriate retrieval strategy, and Ollama generates the final response using retrieved context.

## Tech Stack

**Frontend**
- Angular
- TypeScript

**Backend**
- FastAPI
- LangGraph
- SQLite
- SQLAlchemy

**AI/ML**
- DistilBERT
- Transformers
- PyTorch
- Ollama

## Project Structure

```text
adaptive-rag-ai-platform/
├── frontend/
├── edu-intent-classifier/
├── Adaptive_RAG-Fastapi-Langgraph/
└── README.md
```

## Initial Setup

This section assumes a **completely fresh clone**.

### Prerequisites

- Python **3.11** (recommended)
- Node.js (includes npm)
- Ollama installed locally

### Clone Repository

```bash
git clone https://github.com/HemanthHarish07/adaptive-rag-ai-platform.git
cd adaptive-rag-ai-platform
```

### Create Virtual Environment (Windows)

```powershell
py -3.11 -m venv .venv
```

### Activate Virtual Environment (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install Backend Dependencies (Adaptive RAG Backend)

```powershell
pip install -r Adaptive_RAG-Fastapi-Langgraph/requirements.txt
```

### Install Classifier Dependencies (Intent Classifier)

```powershell
pip install -r edu-intent-classifier/requirements.txt
```

### Install Frontend Dependencies

```powershell
cd frontend
npm install
```

## Running the Project

### Terminal 1 — Intent Classifier

```powershell
cd edu-intent-classifier
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Terminal 2 — Adaptive RAG Backend

> Important: start the backend **from inside** `Adaptive_RAG-Fastapi-Langgraph/` otherwise Python may not resolve the app package.

```powershell
cd Adaptive_RAG-Fastapi-Langgraph
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### Terminal 3 — Angular Frontend

```powershell
cd frontend
npx ng serve
```

## Application URLs

- **Frontend:** http://localhost:4200
- **Classifier:** http://localhost:8000/docs
- **Backend:** http://localhost:8001/docs

## API Endpoints

### Classifier Service

- **POST** `/predict`
- **GET** `/health`

### Adaptive RAG Backend

- **POST** `/query`
- **GET** `/health`
- **POST** `/token`
- **POST** `/register`
- **GET** `/users/me`

## Example Queries

### Theory Query

Input:

```text
What is machine learning?
```

Expected Route:

```text
vectorstore
```

### Debugging Query

Input:

```text
How do I fix a Python traceback error?
```

Expected Route:

```text
websearch
```

## Ollama

**Current verified model:**

- `qwen2.5-coder:3b`

### Optional upgrade

You may optionally pull a larger model:

```bash
ollama pull qwen2.5:7b
```

### Environment variables

- `OLLAMA_PRIMARY_MODEL=qwen2.5:7b`
- `OLLAMA_FALLBACK_MODEL=qwen2.5-coder:3b`

**Note:** The 7B model is optional and improves answer quality but is not required.

## Common Startup Issues

### 1) Backend package import errors

Symptom: backend fails to start due to import/package resolution.

Fix: run the backend from inside:

- `Adaptive_RAG-Fastapi-Langgraph/`

### 2) Port conflicts

Symptom: `address already in use` on one of the ports.

Fix: stop the process using the port or change the host/port in the startup commands.

### 3) Ollama model not available

Symptom: model load failures or missing model errors.

Fix: ensure Ollama is running and that `qwen2.5-coder:3b` is installed (or configure models via the environment variables shown above).

## Resume-Relevant Engineering Concepts

### Key Engineering Highlights

- DistilBERT-based intent classification
- **Adaptive RAG routing** to select retrieval strategies per query
- **LangGraph** orchestration workflow
- **FastAPI** microservices architecture
- **JWT-secured** authentication
- Local LLM inference with **Ollama**
- **Angular + FastAPI** full-stack implementation
- Retrieval transparency and explainability (route + retrieval strategy visibility)

## (ATS) Notes for Recruiters

- Retrieval strategy is chosen dynamically based on intent.
- System exposes routing transparency to support user trust.
- All services run locally using verified endpoints.

## Skills Demonstrated

- Retrieval-Augmented Generation (RAG)
- Natural Language Processing (NLP)
- DistilBERT Intent Classification
- LangGraph Workflow Orchestration
- FastAPI Development
- Angular Development
- JWT Authentication
- Vector Search
- Ollama Integration
- PyTorch
- Transformers
- REST API Design
- Full Stack Development

## Example Response

Query:

What is machine learning?

Output Metadata:

- predicted_intent: theory
- confidence_score: 0.85
- selected_route: vectorstore
- retrieval_strategy: chroma_vector_search

Response:
Generated locally using Ollama.