import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict

# Avoid relative import issues by using absolute package import.
from src.inference import IntentClassifier


from contextlib import asynccontextmanager

# Global reference to classifier
classifier = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler to load the model on startup and clean up resources on shutdown.
    """
    global classifier
    # Resolve model path relative to this file so service works from repo root.
    model_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", "intent_classifier")
    )

    # If the trained model does not exist yet, we will log a warning.
    # The application will fail to load if predicted, but we can catch it gracefully.
    try:
        classifier = IntentClassifier(model_dir)

        print("Model loaded successfully at startup!")
    except Exception as e:
        print(f"Warning: Could not load model at startup: {e}")
        print("FastAPI app will start but `/predict` will raise errors until model is trained.")
        
    yield
    print("Shutting down application...")

app = FastAPI(
    title="Educational Intent Classifier API",
    description="Production FastAPI service for classifying student intent in a RAG routing pipeline.",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic schemas for request/response validation
class QueryRequest(BaseModel):
    query: str = Field(..., description="The student educational query to classify", example="How do I write a binary search tree in Python?")

class BatchQueryRequest(BaseModel):
    queries: List[str] = Field(..., description="List of student queries to classify", example=["What is mitosis?", "Explain how quicksort works."])

class PredictionResponse(BaseModel):
    query: str = Field(..., description="The input query")
    label: str = Field(..., description="The predicted intent class")
    confidence: float = Field(..., description="The confidence score of the prediction (0.0 to 1.0)")
    probabilities: Dict[str, float] = Field(..., description="The probability distribution across all intent classes")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_intent(request: QueryRequest):
    """
    Predicts the intent class of a single student query.
    """
    global classifier
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Ensure that the fine-tuned model exists in the `models/intent_classifier` directory."
        )
    
    try:
        result = classifier.predict(request.query)
        return PredictionResponse(
            query=result["query"],
            label=result["label"],
            confidence=result["confidence"],
            probabilities=result["probabilities"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_intent_batch(request: BatchQueryRequest):
    """
    Predicts the intent classes for a batch of student queries.
    """
    global classifier
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Ensure that the fine-tuned model exists in the `models/intent_classifier` directory."
        )
    
    try:
        results = classifier.predict_batch(request.queries)
        predictions = [
            PredictionResponse(
                query=res["query"],
                label=res["label"],
                confidence=res["confidence"],
                probabilities=res["probabilities"]
            )
            for res in results
        ]
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/health", tags=["System"])
async def health_check():
    """
    Performs a health check on the API and checks if the model is loaded.
    """
    global classifier
    model_status = "loaded" if classifier is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "device": classifier.device if classifier is not None else "unknown"
    }

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    """
    Serves a stunning premium glassmorphic dark-mode web interface for testing the model.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Intent Classifier Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-color: #0b0f19;
                --card-bg: rgba(255, 255, 255, 0.03);
                --card-border: rgba(255, 255, 255, 0.08);
                --text-main: #f3f4f6;
                --text-muted: #9ca3af;
                --accent-primary: #6366f1;
                --accent-secondary: #3b82f6;
                --accent-gradient: linear-gradient(135deg, #6366f1 0%, #3b82f6 100%);
                --glow-color: rgba(99, 102, 241, 0.15);
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }

            body {
                background-color: var(--bg-color);
                color: var(--text-main);
                font-family: 'Outfit', sans-serif;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                overflow-x: hidden;
                background-image: 
                    radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.08) 0%, transparent 40%),
                    radial-gradient(circle at 90% 80%, rgba(59, 130, 246, 0.08) 0%, transparent 40%);
            }

            header {
                padding: 2rem 4rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid var(--card-border);
                backdrop-filter: blur(12px);
                position: sticky;
                top: 0;
                z-index: 100;
            }

            .logo {
                font-size: 1.5rem;
                font-weight: 800;
                background: var(--accent-gradient);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: -0.5px;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .logo::before {
                content: '';
                display: inline-block;
                width: 12px;
                height: 12px;
                background: var(--accent-gradient);
                border-radius: 3px;
                box-shadow: 0 0 10px var(--accent-primary);
            }

            .badge {
                padding: 6px 12px;
                background: rgba(99, 102, 241, 0.1);
                border: 1px solid rgba(99, 102, 241, 0.3);
                border-radius: 9999px;
                font-size: 0.8rem;
                font-weight: 600;
                color: #818cf8;
                display: flex;
                align-items: center;
                gap: 6px;
            }

            .badge-pulse {
                width: 8px;
                height: 8px;
                background: #4ade80;
                border-radius: 50%;
                box-shadow: 0 0 8px #4ade80;
                animation: pulse 2s infinite;
            }

            @keyframes pulse {
                0% { transform: scale(0.9); opacity: 0.8; }
                50% { transform: scale(1.15); opacity: 1; }
                100% { transform: scale(0.9); opacity: 0.8; }
            }

            main {
                flex-grow: 1;
                max-width: 1200px;
                width: 100%;
                margin: 0 auto;
                padding: 3rem 2rem;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 3rem;
                align-items: start;
            }

            @media (max-width: 900px) {
                main {
                    grid-template-columns: 1fr;
                    gap: 2rem;
                }
            }

            .hero-section {
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
            }

            .hero-section h1 {
                font-size: 3rem;
                font-weight: 800;
                line-height: 1.1;
                letter-spacing: -1px;
                background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .hero-section p {
                font-size: 1.1rem;
                color: var(--text-muted);
                line-height: 1.6;
            }

            .card {
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                border-radius: 24px;
                padding: 2.5rem;
                backdrop-filter: blur(20px);
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
                transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
            }

            .card:hover {
                border-color: rgba(99, 102, 241, 0.25);
                box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.05);
            }

            .form-group {
                display: flex;
                flex-direction: column;
                gap: 0.8rem;
                margin-bottom: 1.5rem;
            }

            label {
                font-size: 0.9rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: var(--text-muted);
            }

            textarea {
                width: 100%;
                background: rgba(0, 0, 0, 0.2);
                border: 1px solid var(--card-border);
                border-radius: 14px;
                padding: 1.2rem;
                color: var(--text-main);
                font-family: inherit;
                font-size: 1rem;
                resize: none;
                height: 120px;
                transition: border-color 0.3s ease, box-shadow 0.3s ease;
            }

            textarea:focus {
                outline: none;
                border-color: var(--accent-primary);
                box-shadow: 0 0 15px var(--glow-color);
            }

            button {
                width: 100%;
                background: var(--accent-gradient);
                border: none;
                border-radius: 14px;
                padding: 1rem;
                color: white;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s ease, box-shadow 0.3s ease;
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 8px;
            }

            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 20px rgba(99, 102, 241, 0.4);
            }

            button:active {
                transform: translateY(0);
            }

            .preset-container {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-top: 1rem;
            }

            .preset-btn {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid var(--card-border);
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 0.8rem;
                color: var(--text-muted);
                cursor: pointer;
                transition: all 0.2s ease;
            }

            .preset-btn:hover {
                background: rgba(99, 102, 241, 0.1);
                border-color: rgba(99, 102, 241, 0.3);
                color: var(--text-main);
            }

            /* Result Styles */
            .results-container {
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
            }

            .result-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding-bottom: 1rem;
                border-bottom: 1px solid var(--card-border);
            }

            .result-label {
                font-size: 1.5rem;
                font-weight: 700;
                color: #818cf8;
                text-transform: capitalize;
            }

            .result-confidence {
                font-size: 1.1rem;
                font-weight: 600;
                color: var(--text-main);
            }

            .prob-bars {
                display: flex;
                flex-direction: column;
                gap: 1.2rem;
            }

            .prob-row {
                display: flex;
                flex-direction: column;
                gap: 0.4rem;
            }

            .prob-info {
                display: flex;
                justify-content: space-between;
                font-size: 0.9rem;
            }

            .prob-name {
                text-transform: capitalize;
                font-weight: 500;
                color: var(--text-main);
            }

            .prob-pct {
                font-weight: 600;
                font-family: 'JetBrains Mono', monospace;
                color: var(--accent-secondary);
            }

            .bar-bg {
                width: 100%;
                height: 8px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 99px;
                overflow: hidden;
            }

            .bar-fill {
                height: 100%;
                background: var(--accent-gradient);
                border-radius: 99px;
                width: 0%;
                transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
            }

            .placeholder-text {
                color: var(--text-muted);
                text-align: center;
                padding: 4rem 0;
                font-style: italic;
            }

            .routing-info-card {
                background: rgba(99, 102, 241, 0.03);
                border: 1px dashed rgba(99, 102, 241, 0.2);
                border-radius: 12px;
                padding: 1.2rem;
                margin-top: 1rem;
                font-size: 0.85rem;
                line-height: 1.5;
            }

            .routing-title {
                font-weight: 600;
                color: #818cf8;
                margin-bottom: 4px;
                display: flex;
                align-items: center;
                gap: 6px;
            }

            footer {
                padding: 2rem 4rem;
                border-top: 1px solid var(--card-border);
                text-align: center;
                color: var(--text-muted);
                font-size: 0.85rem;
                backdrop-filter: blur(12px);
            }

            /* Loading spinner */
            .spinner {
                border: 3px solid rgba(255, 255, 255, 0.1);
                width: 20px;
                height: 20px;
                border-radius: 50%;
                border-left-color: white;
                animation: spin 1s linear infinite;
                display: none;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <header>
            <div class="logo">EduIntent Classifier</div>
            <div class="badge">
                <span class="badge-pulse"></span>
                API ACTIVE
            </div>
        </header>

        <main>
            <div class="hero-section">
                <h1>Classifying Intent For RAG Routing</h1>
                <p>
                    Fine-tuned DistilBERT educational classifier designed to serve as the critical gateway in our adaptive LangGraph RAG pipeline. 
                    It maps student queries into structured retrieval domains.
                </p>
                
                <div class="card">
                    <div class="form-group">
                        <label for="query-input">Student Query</label>
                        <textarea id="query-input" placeholder="Type an educational question here..."></textarea>
                    </div>
                    <button onclick="classifyQuery()">
                        <span class="spinner" id="btn-spinner"></span>
                        <span id="btn-text">Classify Intent</span>
                    </button>
                    
                    <div style="margin-top: 1.5rem;">
                        <label>Try Preset Queries:</label>
                        <div class="preset-container">
                            <button class="preset-btn" onclick="setPreset('Explain how polymorphism works in C++.')">concept_explanation</button>
                            <button class="preset-btn" onclick="setPreset('How do I reverse a linked list in Rust?')">coding_help</button>
                            <button class="preset-btn" onclick="setPreset('Give me 5 practice questions on calculus for my exam.')">exam_preparation</button>
                            <button class="preset-btn" onclick="setPreset('Why am I getting a KeyError on line 12 inside my Python script?')">debugging</button>
                            <button class="preset-btn" onclick="setPreset('What does CRUD stand for?')">definition</button>
                            <button class="preset-btn" onclick="setPreset('Derive the formula for the convergence of gradient descent.')">theory</button>
                        </div>
                    </div>
                </div>
            </div>

            <div class="card" style="min-height: 480px; display: flex; flex-direction: column; justify-content: center;">
                <div id="placeholder" class="placeholder-text">
                    Submit a query to view real-time intent classification and routing behavior.
                </div>
                
                <div id="results" class="results-container" style="display: none;">
                    <div class="result-header">
                        <div>
                            <span style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: var(--text-muted); display: block; margin-bottom: 2px;">Predicted Intent</span>
                            <div class="result-label" id="res-label">concept_explanation</div>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: var(--text-muted); display: block; margin-bottom: 2px;">Confidence</span>
                            <div class="result-confidence" id="res-conf">98.42%</div>
                        </div>
                    </div>
                    
                    <div class="prob-bars" id="prob-container">
                        <!-- Bars populated by JS -->
                    </div>

                    <div class="routing-info-card">
                        <div class="routing-title" id="routing-title-text">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18l6-6-6-6"/></svg>
                            Future LangGraph Routing Strategy
                        </div>
                        <span id="routing-desc-text"></span>
                    </div>
                </div>
            </div>
        </main>

        <footer>
            &copy; 2026 AI Learning Platform &bull; Project 3 Intent Classification Subsystem
        </footer>

        <script>
            const routingStrategies = {
                "concept_explanation": {
                    title: "Route: Deep Courseware & Curriculum RAG",
                    desc: "This query will be routed to search academic textbooks and syllabus curriculum indexes. The LLM prompt will trigger a tutorial-style explanation enriched with analogies."
                },
                "coding_help": {
                    title: "Route: Syntax & Code Repo Retriever",
                    desc: "This query will bypass textbooks and target repository indexing or official language documentations. Outputs will be formatted strictly with clean, executable syntax highlighting."
                },
                "exam_preparation": {
                    title: "Route: Assessment Bank & Interactive Quiz Agent",
                    desc: "This will route to our mock question datastore. A LangGraph generator agent will initiate an active recall session, prompting the user with a tailored set of mock questions."
                },
                "debugging": {
                    title: "Route: StackOverflow Index & Code Interpreter Sandbox",
                    desc: "This will route to public coding forums indexing or launch a secure code-execution sandbox to validate code fixes before responding with annotated code diffs."
                },
                "definition": {
                    title: "Route: Glossary Vector Lookup & Flashcard Service",
                    desc: "This query will perform a high-speed direct glossary vector search and generate a short, clean flashcard response designed for fast-paced terminology review."
                },
                "theory": {
                    title: "Route: Semantic Scholar & Academic Paper RAG",
                    desc: "This query will route to academic research paper databases (e.g. arXiv, Semantic Scholar). The system will return a mathematically rigorous derivation or formal citation."
                }
            };

            function setPreset(text) {
                document.getElementById('query-input').value = text;
                classifyQuery();
            }

            async function classifyQuery() {
                const queryText = document.getElementById('query-input').value.trim();
                if (!queryText) return;

                // UI Loading state
                const spinner = document.getElementById('btn-spinner');
                const btnText = document.getElementById('btn-text');
                const placeholder = document.getElementById('placeholder');
                const results = document.getElementById('results');

                spinner.style.display = 'inline-block';
                btnText.textContent = 'Classifying...';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: queryText })
                    });
                    
                    if (!response.ok) {
                        const err = await response.json();
                        throw new Error(err.detail || 'Inference failed');
                    }

                    const data = await response.json();
                    
                    // Show results, hide placeholder
                    placeholder.style.display = 'none';
                    results.style.display = 'flex';
                    
                    // Set header predictions
                    document.getElementById('res-label').textContent = data.label.replace('_', ' ');
                    document.getElementById('res-conf').textContent = (data.confidence * 100).toFixed(2) + '%';
                    
                    // Populating probability bars
                    const container = document.getElementById('prob-container');
                    container.innerHTML = '';
                    
                    // Sort probabilities
                    const sortedProbs = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
                    
                    sortedProbs.forEach(([name, prob]) => {
                        const row = document.createElement('div');
                        row.className = 'prob-row';
                        
                        const pctStr = (prob * 100).toFixed(2) + '%';
                        row.innerHTML = `
                            <div class="prob-info">
                                <span class="prob-name">${name.replace('_', ' ')}</span>
                                <span class="prob-pct">${pctStr}</span>
                            </div>
                            <div class="bar-bg">
                                <div class="bar-fill" style="width: 0%"></div>
                            </div>
                        `;
                        container.appendChild(row);
                        
                        // Force a reflow to trigger CSS transitions
                        setTimeout(() => {
                            row.querySelector('.bar-fill').style.width = pctStr;
                        }, 50);
                    });

                    // Set LangGraph Routing explanation
                    const strategy = routingStrategies[data.label] || { title: "Generic Route", desc: "Route to general knowledge retriever." };
                    document.getElementById('routing-title-text').innerHTML = `
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 6px;"><polyline points="9 18 15 12 9 6"></polyline></svg>
                        ${strategy.title}
                    `;
                    document.getElementById('routing-desc-text').textContent = strategy.desc;

                } catch (error) {
                    alert('Error: ' + error.message + '\\n\\nPlease make sure you have trained the model using `python run.py --train` first.');
                    placeholder.style.display = 'block';
                    results.style.display = 'none';
                } finally {
                    spinner.style.display = 'none';
                    btnText.textContent = 'Classify Intent';
                }
            }
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
