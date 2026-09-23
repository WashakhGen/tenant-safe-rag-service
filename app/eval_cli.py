import asyncio
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from app.evaluation import evaluate_benchmark
else:
    from .evaluation import evaluate_benchmark

if __name__ == "__main__":
    report = asyncio.run(evaluate_benchmark())
    print(json.dumps(report, indent=2, sort_keys=True))
