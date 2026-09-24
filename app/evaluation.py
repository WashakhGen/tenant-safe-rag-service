import re
from typing import TypedDict, cast

from .benchmark import BENCHMARK
from .llm import model_client
from .vector import vector_client


class BenchmarkCase(TypedDict):
    id: str
    tenant: str
    question: str
    relevant_chunk_ids: list[str]
    required_facts: list[str]


class CaseResult(TypedDict):
    id: str
    retrieved_ids: list[str]
    retrieval_recall_at_k: float
    retrieval_mrr: float
    grounded: bool
    tenant_leaks: int


# Lowercase and collapse extra spaces
def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


# Safely reads chunk_id from an item
def _chunk_id(item) -> str | None:
    return getattr(item, "chunk_id", None)


# Safely reads the content text of an item
def _content(item) -> str:
    return getattr(item, "content", None) or str(item)


# Checks that a fact appears as a whole phrase
def _contains_fact(text: str, fact: str) -> bool:
    # whole-phrase match, so "5 calendar days" does not match "15 calendar days"
    pattern = rf"(?<!\w){re.escape(_normalize(fact))}(?!\w)"
    return re.search(pattern, _normalize(text)) is not None


# Splits an answer into sentences and normalizes each
def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [_normalize(p).rstrip(".!? ") for p in parts if p.strip(" .!?")]


def retrieval_recall_at_k(retrieved, relevant_chunk_ids, k=None):

    relevant = set(relevant_chunk_ids)  # the right chunks, as a set
    if not relevant:
        return 0.0

    top = retrieved if k is None else retrieved[:k]
    found = {_chunk_id(item) for item in top} & relevant  # the ones we actually got
    return len(found) / len(relevant)  # share found


def retrieval_mrr(retrieved, relevant_chunk_ids, k=None):

    relevant = set(relevant_chunk_ids)  # the right chunks, as a set
    top = retrieved if k is None else retrieved[:k]

    for index, item in enumerate(top, start=1):  # rank 1, 2, 3...
        if _chunk_id(item) in relevant:
            return 1.0 / index  # 1/rank
    return 0.0  # never found


def answer_is_grounded(answer, evidence, required_facts):
    """True only if the answer is complete and every claim is backed by evidence."""

    if not answer or not evidence:  # No answer or no evidence gives
        return False

    evidence_texts = [_content(item) for item in evidence]

    for fact in required_facts:
        # Every required fact is in the answer
        if not _contains_fact(answer, fact):
            return False  # the answer misses a required fact

        # Every required fact is in the evidence.
        if not any(_contains_fact(text, fact) for text in evidence_texts):
            return False  # the fact is not supported by the retrieved evidence

    # Every sentence in the answer is found in the evidence
    normalized_evidence = [_normalize(text) for text in evidence_texts]
    for sentence in _sentences(answer):
        if not any(sentence in text for text in normalized_evidence):
            return False  # the answer says something the evidence does not

    return True


async def evaluate_benchmark(top_k: int = 3):
    rows: list[CaseResult] = []
    for case in cast(list[BenchmarkCase], BENCHMARK):
        retrieved = await vector_client.search(
            case["question"], case["tenant"], top_k
        )  # get chunks
        answer = await model_client.answer(
            case["question"], retrieved
        )  # build the answer
        rows.append(
            {
                "id": case["id"],
                "retrieved_ids": [
                    item.chunk_id for item in retrieved
                ],  # what we found (for debugging)
                "retrieval_recall_at_k": retrieval_recall_at_k(  # the three scores
                    retrieved, case["relevant_chunk_ids"], top_k
                ),
                "retrieval_mrr": retrieval_mrr(
                    retrieved, case["relevant_chunk_ids"], top_k
                ),
                "grounded": answer_is_grounded(
                    answer, retrieved, case["required_facts"]
                ),
                "tenant_leaks": sum(  # any chunk from the wrong tenant?
                    item.account_id != case["tenant"] for item in retrieved
                ),
            }
        )

    n = len(rows) or 1  # "or 1" avoids dividing by zero if the benchmark is empty

    return {
        "n_cases": len(rows),
        "k": top_k,
        "retrieval_recall_at_k": sum(r["retrieval_recall_at_k"] for r in rows)
        / n,  # average recall
        "retrieval_mrr": sum(r["retrieval_mrr"] for r in rows) / n,  # average MRR
        "grounded_answer_rate": sum(r["grounded"] for r in rows)
        / n,  #  share of cases that were grounded
        "tenant_leaks": sum(r["tenant_leaks"] for r in rows),  # total leaks,
        "cases": rows,  # the per-question detail
    }
