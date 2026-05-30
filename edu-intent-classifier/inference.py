import os
import sys

# Configure sys.path so src is accessible
project_root = os.path.dirname(os.path.abspath(__file__))
if os.path.join(project_root, "src") not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.inference import main

if __name__ == "__main__":
    main()
