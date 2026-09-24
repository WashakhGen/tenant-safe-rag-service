import pytest

from app.evaluation import retrieval_mrr, retrieval_recall_at_k
from app.schemas import Evidence


def _retrieved(*chunk_ids):
    return [
        Evidence(chunk_id=i, account_id="atlas", content=f"text of {i}", score=1.0)
        for i in chunk_ids
    ]


@pytest.mark.parametrize(
    ("retrieved", "relevant", "expected"),
    [
        (["a", "b", "c"], ["a"], 1.0),
        (["a", "b", "c"], ["c"], 1.0),
        (["a", "b", "c"], ["a", "c"], 1.0),
        (["a", "b", "c"], ["a", "x"], 0.5),
        (["a", "b", "c"], ["x"], 0.0),
        (["a", "b", "c"], ["x", "y", "a", "z"], 0.25),
        ([], ["a"], 0.0),
        (["a", "b"], [], 0.0),
        (["a", "a", "a"], ["a", "b"], 0.5),  # repeats must not inflate recall
    ],
)
def test_recall_is_the_share_of_relevant_chunks_found(retrieved, relevant, expected):
    assert retrieval_recall_at_k(_retrieved(*retrieved), relevant) == expected


def test_recall_ignores_irrelevant_evidence():
    assert retrieval_recall_at_k(_retrieved("wrong"), ["good"]) == 0.0


def test_recall_only_looks_at_the_top_k():
    retrieved = _retrieved("a", "b", "c")
    assert retrieval_recall_at_k(retrieved, ["c"], k=2) == 0.0
    assert retrieval_recall_at_k(retrieved, ["c"], k=3) == 1.0


@pytest.mark.parametrize(
    ("retrieved", "relevant", "expected"),
    [
        (["a", "b", "c"], ["a"], 1.0),
        (["x", "a", "c"], ["a"], 0.5),
        (["x", "y", "a"], ["a"], 1 / 3),
        (["x", "y", "z"], ["a"], 0.0),
        ([], ["a"], 0.0),
        (["x", "b", "a"], ["a", "b"], 0.5),  # only the first relevant hit counts
        (["a", "a"], ["a"], 1.0),
    ],
)
def test_mrr_depends_on_the_position_of_the_first_relevant_chunk(
    retrieved, relevant, expected
):
    assert retrieval_mrr(_retrieved(*retrieved), relevant) == pytest.approx(expected)


def test_mrr_rewards_earlier_ranking():
    early = retrieval_mrr(_retrieved("a", "x", "y"), ["a"])
    late = retrieval_mrr(_retrieved("x", "y", "a"), ["a"])
    assert early > late


def test_mrr_only_looks_at_the_top_k():
    assert retrieval_mrr(_retrieved("x", "y", "a"), ["a"], k=2) == 0.0


def test_recall_and_mrr_measure_different_things():
    # both relevant chunks found (recall 1.0) but the first one sits at rank 2
    retrieved = _retrieved("x", "a", "b")
    assert retrieval_recall_at_k(retrieved, ["a", "b"]) == 1.0
    assert retrieval_mrr(retrieved, ["a", "b"]) == 0.5
