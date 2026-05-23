import os
import json
import time
import random
import pandas as pd
from typing import List, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define intent classes, descriptions, and offline templates
INTENT_CLASSES = {
    "concept_explanation": {
        "description": "Deep educational queries asking for detailed explanations of concepts, algorithms, processes, or mechanisms. The user wants to understand 'how it works' or 'why it behaves this way' beyond a simple definition.",
        "offline_templates": [
            "Explain how {topic} works in simple terms.",
            "Can you explain the mechanics behind {topic}?",
            "How does {topic} actually work under the hood?",
            "Describe the step-by-step process of {topic}.",
            "I'm struggling to understand the difference between {topic} and {topic2}. Can you clarify?",
            "Give me a deep dive explanation of {topic}.",
            "How do {topic} and {topic2} interact with each other?",
            "Why is {topic} considered a fundamental process in {subject}?",
            "Explain the concept of {topic} to a 10-year-old.",
            "Can you walk me through the intuition behind {topic}?",
            "What happens during {topic} and why does it occur?",
            "Explain the physical process of {topic} in detail.",
            "How does the mechanism of {topic} affect {topic2}?",
            "What is the underlying concept of {topic} and why is it important?",
            "Can you explain {topic} using a real-world analogy?"
        ],
        "topics": {
            "computer_science": ["quicksort", "memory management", "garbage collection", "REST APIs", "asynchronous programming", "database indexing"],
            "mathematics": ["matrix multiplication", "limit of a function", "eigenvectors", "Bayes' theorem", "derivatives", "Fourier transform"],
            "science": ["photosynthesis", "cellular respiration", "mitosis", "plate tectonics", "nuclear fusion", "natural selection"],
            "economics": ["inflation", "supply and demand curve", "fractional reserve banking", "fiscal policy", "opportunity cost", "game theory"]
        }
    },
    "coding_help": {
        "description": "Queries asking for code snippets, syntax implementation, boilerplate, or programming guidance. The user wants to know how to write code to achieve a specific task in any programming language.",
        "offline_templates": [
            "How do I implement a {data_structure} in {language}?",
            "Write a {language} script to {task}.",
            "Can you show me the syntax for {task} in {language}?",
            "Give me a clean {language} implementation of {algorithm}.",
            "How to write a function in {language} that {task}?",
            "What is the best way to handle {task} in {language}?",
            "Show me a simple example of using {library} for {task}.",
            "Write a boilerplate for a {language} application that connects to {service}.",
            "How do I loop through a {data_structure} and {task} in {language}?",
            "Can you help me write code for {algorithm} in {language}?",
            "How to parse {data_format} in {language}?",
            "Write a SQL query to {sql_task}.",
            "Show me the standard library function for {task} in {language}.",
            "How do you define a class with a custom constructor in {language}?",
            "Write an asynchronous function in {language} to fetch data from an API."
        ],
        "variables": {
            "data_structure": ["binary search tree", "linked list", "hash map", "priority queue", "graph", "trie"],
            "language": ["Python", "JavaScript", "Java", "C++", "Rust", "Go", "TypeScript"],
            "task": ["read a CSV file", "make an HTTP POST request", "validate an email address", "sort an array of dictionaries", "calculate the fibonacci sequence", "hash a password"],
            "algorithm": ["binary search", "dijkstra's algorithm", "merge sort", "bubble sort", "breadth-first search", "depth-first search"],
            "library": ["pandas", "express.js", "requests", "numpy", "react", "spring boot"],
            "service": ["a PostgreSQL database", "a Redis cache", "an external REST API", "MongoDB", "AWS S3"],
            "data_format": ["a JSON string", "a CSV file", "an XML document", "a YAML config"],
            "sql_task": ["select employees with salary greater than 50000", "perform an inner join between users and orders", "group sales by month and calculate the average", "find duplicate records in a table"]
        }
    },
    "exam_preparation": {
        "description": "Queries asking for practice questions, quiz generation, flashcards, study guides, review plans, or exam strategies for a particular topic or subject.",
        "offline_templates": [
            "Give me 5 practice questions about {topic} for my exam.",
            "Generate a practice quiz on {topic} with answers.",
            "Can you help me prepare for a test on {topic}?",
            "What are the key concepts I need to study for an exam on {topic}?",
            "Create a study guide and study plan for {topic}.",
            "Give me some flashcard prompts for active recall on {topic}.",
            "What are the most common exam questions about {topic}?",
            "Quiz me on {topic}. Ask me one question at a time.",
            "I have an exam on {topic} tomorrow, what are the high-yield topics?",
            "Generate some multiple-choice questions about {topic}.",
            "Explain how to solve a standard exam problem on {topic}.",
            "Create a comprehensive cheat sheet for my {subject} final exam covering {topic}.",
            "Can you review these study notes on {topic} and tell me what is missing?",
            "Give me a mock exam for AP {subject} with a focus on {topic}.",
            "Provide some practice problems on {topic} with step-by-step solutions."
        ],
        "topics": [
            "linear algebra", "calculus derivative rules", "organic chemistry reactions", "Newton's laws of motion",
            "microeconomics principles", "the French Revolution", "cellular respiration", "object-oriented programming",
            "operating systems scheduling", "probability and statistics", "molecular biology", "macroeconomic policies"
        ],
        "subject": ["Computer Science", "Physics", "Chemistry", "Biology", "Calculus", "History", "Economics"]
    },
    "debugging": {
        "description": "Queries involving broken code, runtime errors, compile-time exceptions, unexpected output, or logic bugs. The user provides code or an error message and asks for help fixing it.",
        "offline_templates": [
            "Why am I getting a {error_type} in this {language} code?",
            "Fix this {language} code: it is throwing a {error_type}.",
            "Can you help me debug this {language} function? It returns the wrong output.",
            "Why does my {language} program raise a {error_type} when I run it?",
            "I have a bug in my {language} code. Here is the snippet: {code_snippet}",
            "My {language} loop runs infinitely. How do I fix it?",
            "How do I resolve a {error_type} in {library}?",
            "Explain this error message: {error_msg}",
            "Why is my variable returning {null_val} instead of the expected value in {language}?",
            "My {language} code is slow and freezing. Can you look at it for performance issues?",
            "Why does this {language} block throw an exception: {code_snippet}",
            "Help me fix this syntax error in my {language} file.",
            "How do I debug a segmentation fault in my {language} program?",
            "I am getting a {error_type} when trying to parse {data_format} in {language}. How do I solve this?",
            "Why does this SQL query fail: {sql_error}"
        ],
        "variables": {
            "error_type": ["IndexOutOfBoundsException", "NullPointerException", "KeyError", "TypeError", "AttributeError", "ZeroDivisionError", "SyntaxError", "ValueError"],
            "language": ["Python", "JavaScript", "Java", "C++", "Rust", "TypeScript", "SQL"],
            "library": ["pandas", "numpy", "React", "NodeJS", "Spring Boot", "Django"],
            "code_snippet": [
                "def process(items): return items[10] # but items has length 5",
                "int* p = NULL; *p = 10;",
                "my_dict = {'a': 1}; print(my_dict['b'])",
                "const data = null; console.log(data.name);"
            ],
            "error_msg": [
                "Uncaught TypeError: Cannot read properties of undefined (reading 'map')",
                "ValueError: math domain error",
                "django.db.utils.OperationalError: no such table",
                "java.lang.NullPointerException: Cannot invoke 'String.length()' because 'str' is null"
            ],
            "null_val": ["None", "undefined", "null", "NaN"],
            "data_format": ["JSON", "XML", "CSV", "YAML"],
            "sql_error": [
                "SELECT * FROM users WHERE id = 'abc' (id is integer)",
                "SELECT name, SUM(price) FROM orders (missing GROUP BY)",
                "INSERT INTO users (id) VALUES (1) (violates unique constraint)"
            ]
        }
    },
    "definition": {
        "description": "Brief, precise queries seeking direct dictionary-like definitions, terminology clarification, or acronym meanings. The user wants a quick 'what is this' response.",
        "offline_templates": [
            "What is {term}?",
            "Define {term} in simple terms.",
            "What does {acronym} stand for?",
            "Give me a quick definition of {term}.",
            "What is the meaning of {term} in {field}?",
            "Explain what {term} means.",
            "What does the term {term} mean in {field}?",
            "Provide a one-sentence definition of {term}.",
            "What is a {term}?",
            "Can you give me a brief explanation of the term {term}?",
            "What does {acronym} mean in {field}?",
            "Definition of {term}.",
            "What is the difference between {term} and {term2} in terms of definition?",
            "How is {term} defined in {field}?",
            "What exactly is {term}?"
        ],
        "variables": {
            "term": ["polymorphism", "recursion", "mitosis", "inflation", "entropy", "gradient descent", "syntactic sugar", "photosynthesis", "osmosis", "liquidity", "eigenvector", "epistemology", "neuroplasticity", "superconductivity", "quantum superposition"],
            "term2": ["inheritance", "iteration", "meiosis", "deflation", "enthalpy", "gradient ascent", "boilerplate", "cellular respiration", "diffusion", "solvency", "eigenvalue", "ontology", "synaptic pruning", "semiconductivity", "quantum entanglement"],
            "acronym": ["REST", "JSON", "SQL", "API", "CPU", "RAM", "HTTP", "HTML", "CSS", "DNS", "MVC", "OOP", "SDK", "IDE", "GPU"],
            "field": ["Computer Science", "Biology", "Economics", "Physics", "Chemistry", "Mathematics", "Philosophy", "Neuroscience"],
        }
    },
    "theory": {
        "description": "Theoretical, historical, academic, or mathematical queries. This includes requests for mathematical proofs, history of discoveries, academic theory, research papers, or deep academic background.",
        "offline_templates": [
            "What is the mathematical proof of {theory}?",
            "Who discovered {discovery} and what was the historical context?",
            "What is the academic theory behind {theory}?",
            "Can you explain the theoretical foundations of {theory}?",
            "Provide the formal derivation of {theory}.",
            "What is the history of the {discovery}?",
            "What are the core axioms of {theory}?",
            "Explain the theoretical limitations of {theory}.",
            "Can you summarize the original paper on {paper}?",
            "What is the mathematical formulation of {theory}?",
            "What is the debate between {theory} and {theory2} in academic literature?",
            "Who proposed the {theory} and why?",
            "Derive the formula for {formula}.",
            "What is the philosophical or theoretical basis of {theory}?",
            "Explain the proof of {theory} in detail."
        ],
        "variables": {
            "theory": ["the halting problem", "Euler's identity", "the central limit theorem", "Keynesian economics", "special relativity", "the standard model of particle physics", "computational complexity theory", "the P vs NP problem", "Godel's incompleteness theorems", "the Schrödinger equation"],
            "theory2": ["the Church-Turing thesis", "the law of large numbers", "classical economics", "general relativity", "quantum mechanics", "information theory", "algorithmic information theory", "decidability", "Turing completeness"],
            "discovery": ["the structure of DNA", "superconductivity", "the electron", "calculus", "plate tectonics", "the expanding universe", "penicillin", "the photoelectric effect"],
            "paper": ["Attention Is All You Need", "A Mathematical Theory of Communication", "Computing Machinery and Intelligence", "ImageNet Classification with Deep Convolutional Neural Networks"],
            "formula": ["quadratic equation", "Gaussian distribution", "gravitational force", "ideal gas law", "entropy in information theory", "least squares regression coefficient"]
        }
    }
}

def generate_offline_dataset(num_per_class: int = 150) -> pd.DataFrame:
    """
    Generates a highly diverse dataset offline using templating and random combinations.
    """
    print(f"Generating offline dataset with {num_per_class} examples per class...")
    data = []
    
    for label, config in INTENT_CLASSES.items():
        templates = config["offline_templates"]
        generated = set()
        
        # We want to generate num_per_class unique queries
        attempts = 0
        max_attempts = num_per_class * 20
        
        while len(generated) < num_per_class and attempts < max_attempts:
            attempts += 1
            template = random.choice(templates)
            
            # Fill placeholders
            query = template
            
            if label == "concept_explanation":
                topics_dict = config["topics"]
                subj = random.choice(list(topics_dict.keys()))
                t1 = random.choice(topics_dict[subj])
                t2 = random.choice(topics_dict[subj])
                while t1 == t2:
                    t2 = random.choice(topics_dict[subj])
                query = query.format(topic=t1, topic2=t2, subject=subj)
                
            elif label == "coding_help":
                vars_dict = config["variables"]
                ds = random.choice(vars_dict["data_structure"])
                lang = random.choice(vars_dict["language"])
                task = random.choice(vars_dict["task"])
                alg = random.choice(vars_dict["algorithm"])
                lib = random.choice(vars_dict["library"])
                serv = random.choice(vars_dict["service"])
                fmt = random.choice(vars_dict["data_format"])
                sql = random.choice(vars_dict["sql_task"])
                
                query = query.format(
                    data_structure=ds, language=lang, task=task,
                    algorithm=alg, library=lib, service=serv,
                    data_format=fmt, sql_task=sql
                )
                
            elif label == "exam_preparation":
                topic = random.choice(config["topics"])
                subject = random.choice(config["subject"])
                query = query.format(topic=topic, subject=subject)
                
            elif label == "debugging":
                vars_dict = config["variables"]
                err = random.choice(vars_dict["error_type"])
                lang = random.choice(vars_dict["language"])
                lib = random.choice(vars_dict["library"])
                code = random.choice(vars_dict["code_snippet"])
                msg = random.choice(vars_dict["error_msg"])
                nv = random.choice(vars_dict["null_val"])
                fmt = random.choice(vars_dict["data_format"])
                sql_err = random.choice(vars_dict["sql_error"])
                
                query = query.format(
                    error_type=err, language=lang, library=lib,
                    code_snippet=code, error_msg=msg, null_val=nv,
                    data_format=fmt, sql_error=sql_err
                )
                
            elif label == "definition":
                vars_dict = config["variables"]
                term = random.choice(vars_dict["term"])
                term2 = random.choice(vars_dict["term2"])
                while term == term2:
                    term2 = random.choice(vars_dict["term2"])
                acr = random.choice(vars_dict["acronym"])
                fld = random.choice(vars_dict["field"])
                
                query = query.format(
                    term=term, term2=term2, acronym=acr, field=fld
                )
                
            elif label == "theory":
                vars_dict = config["variables"]
                th = random.choice(vars_dict["theory"])
                th2 = random.choice(vars_dict["theory2"])
                while th == th2:
                    th2 = random.choice(vars_dict["theory2"])
                disc = random.choice(vars_dict["discovery"])
                paper = random.choice(vars_dict["paper"])
                form = random.choice(vars_dict["formula"])
                
                query = query.format(
                    theory=th, theory2=th2, discovery=disc,
                    paper=paper, formula=form
                )
            
            # Clean whitespaces
            query = " ".join(query.split())
            generated.add(query)
            
        print(f"  Class '{label}': Generated {len(generated)} queries.")
        for q in generated:
            data.append({"query": q, "label": label})
            
    return pd.DataFrame(data)

def generate_online_dataset_with_gemini(num_per_class: int = 150) -> pd.DataFrame:
    """
    Generates educational queries using Google's Gemini API.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    
    print("Initializing Gemini Client...")
    genai.configure(api_key=api_key)
    
    # We will use gemini-pro as it is supported in this library version
    model = genai.GenerativeModel('gemini-pro')
    
    data = []
    
    for label, config in INTENT_CLASSES.items():
        print(f"Generating queries for class '{label}' via Gemini API...")
        class_queries = []
        
        # We will request in chunks of 50 to avoid token limits and stay highly structured
        chunk_size = 50
        num_chunks = int(num_per_class / chunk_size)
        if num_per_class % chunk_size != 0:
            num_chunks += 1
            
        for chunk in range(num_chunks):
            # Calculate remaining queries needed
            needed = num_per_class - len(class_queries)
            current_chunk_size = min(chunk_size, needed)
            if current_chunk_size <= 0:
                break
                
            print(f"  Requesting chunk {chunk + 1}/{num_chunks} ({current_chunk_size} queries)...")
            
            prompt = f"""You are an expert curriculum designer and AI learning assistant. Your task is to generate high-quality, realistic student queries for a specific educational intent class in an online learning platform.

The class is: "{label}"
Description: {config['description']}

Generate exactly {current_chunk_size} diverse, realistic student queries that fall under this category.
Guidelines:
1. Vary the subjects: include computer science, mathematics, statistics, history, physics, biology, chemistry, linguistics, literature, philosophy, geography, and economics.
2. Vary the tone, vocabulary, and length: some queries should be very brief and casual, some should be long, detailed and conversational, and some should be formal academic requests.
3. Vary the user profiles: some sound like complete novices, some like high school students, some like college students, and some like professional developers or PhD researchers.
4. Avoid repetitive sentence structures. Do not start every query with the same verbs (like "How do I..." or "What is..."). Make some queries look like search terms, some like direct questions to a tutor, and some like a block of pasted text with a prompt.
5. Do NOT include any meta-commentary, lists headers, or numbers.

Provide the output in a clean, valid JSON format containing a single root object with the key "queries" that maps to a list of strings:
{{
  "queries": [
    "example query 1",
    "example query 2"
  ]
}}

Ensure that you return ONLY valid JSON. Do not wrap it in markdown formatting (like ```json). Just return the raw JSON string."""

            response_received = False
            retries = 3
            while retries > 0 and not response_received:
                try:
                    # Generate content directly (relying on prompt for JSON structure)
                    response = model.generate_content(prompt)
                    text = response.text.strip()
                    
                    # Parse JSON safely
                    # Strip any possible markdown fence wrappers
                    if text.startswith("```json"):
                        text = text[7:]
                    if text.endswith("```"):
                        text = text[:-3]
                    text = text.strip()
                    
                    parsed = json.loads(text)
                    queries = parsed.get("queries", [])
                    
                    if not isinstance(queries, list) or len(queries) == 0:
                        raise ValueError("Parsed JSON does not contain a valid list of queries")
                        
                    # Filter and clean
                    cleaned_queries = []
                    for q in queries:
                        q_clean = " ".join(str(q).split())
                        if q_clean and len(q_clean) > 5:
                            cleaned_queries.append(q_clean)
                            
                    class_queries.extend(cleaned_queries)
                    print(f"    Successfully generated {len(cleaned_queries)} queries in this chunk. Total for {label}: {len(class_queries)}/{num_per_class}")
                    response_received = True
                except Exception as e:
                    print(f"    Error in Gemini API generation: {e}. Retrying after sleep...")
                    retries -= 1
                    time.sleep(3)
                    
            if not response_received:
                print(f"    Failed to generate chunk {chunk + 1} after retries. Falling back to templates for this chunk.")
                # Generate a temporary fallback chunk
                fallback_df = generate_offline_dataset(num_per_class=current_chunk_size)
                fallback_queries = fallback_df[fallback_df["label"] == label]["query"].tolist()
                class_queries.extend(fallback_queries)
                
            # Add a slight delay to respect rate limits
            time.sleep(1.5)
            
        # Ensure we have exactly num_per_class queries
        class_queries = list(set(class_queries))[:num_per_class]
        # If we fell short due to set deduplication, fill up with offline templates
        if len(class_queries) < num_per_class:
            extra_needed = num_per_class - len(class_queries)
            fallback_df = generate_offline_dataset(num_per_class=extra_needed * 2)
            fallback_queries = fallback_df[fallback_df["label"] == label]["query"].tolist()
            class_queries.extend(fallback_queries[:extra_needed])
            
        for q in class_queries:
            data.append({"query": q, "label": label})
            
    return pd.DataFrame(data)

def main():
    print("="*60)
    print("EDUCATIONAL INTENT CLASSIFIER - DATA GENERATION STAGE")
    print("="*60)
    
    # Create raw data folder if not exists
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    df = None
    
    if api_key:
        print("GEMINI_API_KEY found! Testing API connection...")
        try:
            genai.configure(api_key=api_key)
            test_model = genai.GenerativeModel('gemini-pro')
            # Test generate content with strict 1 token limit for instant response
            test_model.generate_content("Ping", generation_config={"max_output_tokens": 5})
            print("Gemini API connection test passed! Proceeding to online dataset generation...")
            df = generate_online_dataset_with_gemini(num_per_class=150)
            print("Gemini API dataset generation completed successfully!")
        except Exception as e:
            print(f"Gemini API connection test failed: {e}")
            print("Falling back immediately to high-speed offline template generation...")
            df = generate_offline_dataset(num_per_class=150)
    else:
        print("GEMINI_API_KEY NOT found in environment.")
        print("Executing offline template generation...")
        df = generate_offline_dataset(num_per_class=150)
        
    # Check dataset size and distribution
    print("\nDataset Summary:")
    print(f"Total Examples: {len(df)}")
    print("Class Distribution:")
    print(df["label"].value_counts())
    
    # Save in CSV and JSONL formats
    csv_path = "data/raw/intent_dataset.csv"
    jsonl_path = "data/raw/intent_dataset.jsonl"
    
    # Save CSV
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nSaved CSV dataset to: {os.path.abspath(csv_path)}")
    
    # Save JSONL
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    print(f"Saved JSONL dataset to: {os.path.abspath(jsonl_path)}")
    
    print("="*60)

if __name__ == "__main__":
    main()
