"""Development-only wrapper for environments where dependencies live in .deps."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / ".deps"), str(ROOT)]

from bench.run_benchmark import main

if __name__ == "__main__":
    main()
