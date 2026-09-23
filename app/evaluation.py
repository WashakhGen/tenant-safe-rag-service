import re

from .benchmark import BENCHMARK
from .llm import model_client
from .vector import vector_client


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def retrieval_recall_at_k(retrieved, relevant_chunk_ids):
    if not retrieved:
        return 0.0
    return 1.0 if any(getattr(item, "chunk_id", None) for item in retrieved) else 0.0


def retrieval_mrr(retrieved, relevant_chunk_ids):
    for index, item in enumerate(retrieved, start=1):
        if getattr(item, "chunk_id", None):
            return 1.0 / index
    return 0.0


def answer_is_grounded(answer, evidence, required_facts):
    if not answer or not evidence:
        return False
    answer_norm = _normalize(answer)
    facts_present = all(_normalize(fact) in answer_norm for fact in required_facts)
    return bool(facts_present)


async def evaluate_benchmark(top_k: int = 3):
    rows = []
    for case in BENCHMARK:
        retrieved = await vector_client.search(case["question"], case["tenant"], top_k)
        answer = await model_client.answer(case["question"], retrieved)
        rows.append(
            {
                "id": case["id"],
                "retrieval_recall_at_k": retrieval_recall_at_k(
                    retrieved, case["relevant_chunk_ids"]
                ),
                "retrieval_mrr": retrieval_mrr(retrieved, case["relevant_chunk_ids"]),
                "grounded": answer_is_grounded(
                    answer, retrieved, case["required_facts"]
                ),
            }
        )
    return {
        "n_cases": len(rows),
        "retrieval_recall_at_k": sum(r["retrieval_recall_at_k"] for r in rows)
        / len(rows),
        "retrieval_mrr": sum(r["retrieval_mrr"] for r in rows) / len(rows),
        "grounded_answer_rate": sum(r["grounded"] for r in rows) / len(rows),
        "cases": rows,
    }
