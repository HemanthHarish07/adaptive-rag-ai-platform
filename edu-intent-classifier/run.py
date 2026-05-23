import os
import sys
import argparse
import subprocess

def run_command(command: list, description: str):
    """
    Helper function to run subprocess commands with clean console logging.
    """
    print("="*60)
    print(f"Executing: {description}")
    print(f"Command: {' '.join(command)}")
    print("="*60)
    
    try:
        # Run command and pipe output live to the screen
        process = subprocess.Popen(
            command,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True
        )
        process.wait()
        
        if process.returncode != 0:
            print(f"\n[ERROR] Command failed with exit code {process.returncode}")
            sys.exit(process.returncode)
            
    except KeyboardInterrupt:
        print("\n[INFO] Process interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Failed to execute process: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Master Orchestration CLI for the Educational Intent Classifier Subsystem"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", action="store_true", help="Generate/regenerate the synthetic training dataset")
    group.add_argument("--train", action="store_true", help="Run the model fine-tuning process")
    group.add_argument("--evaluate", action="store_true", help="Evaluate the fine-tuned model and plot confusion matrix")
    group.add_argument("--inference", action="store_true", help="Launch interactive CLI inference tester")
    group.add_argument("--serve", action="store_true", help="Start the production FastAPI inference microservice")
    
    # Optional arguments to pass down
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Base transformer model name (for --train)")
    parser.add_argument("--epochs", type=str, default="3", help="Number of epochs to train (for --train)")
    parser.add_argument("--batch_size", type=str, default="16", help="Batch size for training/evaluation (for --train)")
    parser.add_argument("--port", type=str, default="8000", help="Port to run FastAPI server (for --serve)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run FastAPI server (for --serve)")
    
    args = parser.parse_args()
    
    # Change working directory to the project root to ensure relative paths resolve correctly
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # Avoid PYTHONPATH hacks. Keep sys.path minimal.
    # Import resolution will rely on running uvicorn with the correct module path.
    sys.path.append(project_root)

    
    # Force HuggingFace to use PyTorch backend and bypass TensorFlow/Keras checks
    os.environ["USE_TF"] = "NO"
    os.environ["USE_TORCH"] = "YES"
    
    if args.generate:
        run_command(
            [sys.executable, "src/data_generation.py"],
            "Synthetic Dataset Generation (Gemini API / Scripted fallback)"
        )
        
    elif args.train:
        run_command(
            [
                sys.executable, "src/train.py",
                "--model_name", args.model_name,
                "--epochs", args.epochs,
                "--batch_size", args.batch_size
            ],
            "Fine-Tuning Base Transformer"
        )
        
    elif args.evaluate:
        run_command(
            [sys.executable, "src/evaluate.py"],
            "Model Evaluation and Confusion Matrix Generation"
        )
        
    elif args.inference:
        # We run it interactively (so we need subprocess)
        run_command(
            [sys.executable, "src/inference.py"],
            "Interactive CLI Inference Playground"
        )
        
    elif args.serve:
        print("="*60)
        print("Starting FastAPI Microservice...")
        print(f"URL: http://localhost:{args.port}")
        print("="*60)

        # Start uvicorn server.
        # Use the fully-qualified import path to avoid ambiguity.
        # Run without --reload to avoid subprocess reload edge-cases.
        # Ensure `src` package is importable by uvicorn.
        # uvicorn runs in a separate process; we must set PYTHONPATH explicitly.
        old_pp = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = os.path.join(project_root, "edu-intent-classifier") + (
            os.pathsep + old_pp if old_pp else ""
        )

        run_command(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app:app",

                "--host",
                args.host,
                "--port",
                args.port,
            ],
            "FastAPI Server",
        )




if __name__ == "__main__":
    main()

