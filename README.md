# Dvxel AI Engineer Assessment - Variant B

This is the candidate starter repository for the Dvxel AI Engineer technical assessment. The service is intentionally small and contains production defects across a multi-tenant RAG path plus an evaluation harness.

## Setup

Python 3.11+ is recommended. From the repository root:

```text
python -m venv .venv
python -m pip install -r requirements.txt
pytest -q
```

The starter includes baseline tests plus three AI-focused tests that are expected to fail before you begin. They mark the AI-specific portion of the exercise: retrieval measurement, grounding measurement, and multi-fact context assembly.

## Task

Improve the existing service without rebuilding it from scratch. Your implementation should be safe for multi-tenant production use and should preserve the existing API shape unless a change is necessary.

### A. Production AI service correctness

- trusted tenant identity and strict tenant isolation
- retrieval constrained to the trusted tenant and ranked by the query rather than fixed storage order
- request isolation under concurrent traffic
- bounded timeout/retry behavior and controlled dependency failure semantics
- focused automated tests for security, retrieval, concurrency, and failure paths

### B. AI/RAG evaluation

The repository contains a small local benchmark in `app/benchmark.py` and an intentionally flawed evaluator in `app/evaluation.py`. Make the evaluation meaningful. At minimum:

- compute retrieval Recall@k against the provided relevant chunk IDs
- compute MRR so the position of the first relevant chunk matters
- compute grounded answer rate separately from retrieval success
- use the provided ground-truth facts and retrieved evidence; an answer with missing or unsupported facts must not count as grounded
- keep the benchmark deterministic and runnable without external model or vector credentials

Add tests for both retrieval metric correctness and generation/grounding correctness.

### C. Documentation

Add `DECISIONS.md` (400 words maximum) describing your trust boundary, retrieval/evaluation approach, failure semantics, and one trade-off.

If you used AI tools during the take-home, add `AI_USAGE.md` with a brief description of where they materially contributed and one concrete example of an output you corrected or rejected.

## Constraints

Do not replace the project with a framework demo or external service. No AWS, Pinecone, or model credentials are required. Keep the existing endpoint working unless a change is necessary for correctness.

The evaluator will run additional tests that are not included in the repository.

## Submission

Return the complete repository with your changes, `DECISIONS.md`, the tests you added or changed, and `AI_USAGE.md` when applicable.
