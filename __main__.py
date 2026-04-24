"""Entry point: python -m . --config configs/base.yaml"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.run_all import main

if __name__ == "__main__":
    main()
