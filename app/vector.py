import asyncio
import re

from .schemas import Evidence

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "am",
        "be",
        "can",
        "do",
        "does",
        "for",
        "from",
        "give",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "of",
        "on",
        "or",
        "our",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
    ]
)


def _stem(word: str) -> str:
    # crude plural handling so "refunds" matches "refund"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _terms(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {_stem(w) for w in words if w not in _STOPWORDS}


def _doc_terms(doc: Evidence) -> set[str]:
    keywords = " ".join(doc.metadata.get("keywords", []))
    topic = doc.metadata.get("topic", "")
    return _terms(f"{doc.content} {keywords} {topic}")


class VectorStoreClient:
    def __init__(self):
        self.docs = {
            "atlas": [
                {
                    "chunk_id": "at1",
                    "account_id": "atlas",
                    "content": "Refunds are allowed within 5 calendar days of capture.",
                    "metadata": {
                        "keywords": ["refund", "returns", "capture"],
                        "topic": "refund",
                    },
                    "score": 0.95,
                },
                {
                    "chunk_id": "at2",
                    "account_id": "atlas",
                    "content": "Settlement files are published at 01:30 UTC each day.",
                    "metadata": {
                        "keywords": ["settlement", "files", "daily"],
                        "topic": "settlement",
                    },
                    "score": 0.83,
                },
                {
                    "chunk_id": "at3",
                    "account_id": "atlas",
                    "content": "Chargeback evidence is uploaded through the disputes console.",
                    "metadata": {
                        "keywords": ["chargeback", "evidence", "disputes"],
                        "topic": "chargeback",
                    },
                    "score": 0.7,
                },
            ],
            "cedar": [
                {
                    "chunk_id": "cd1",
                    "account_id": "cedar",
                    "content": "Refunds are allowed within 10 calendar days of capture.",
                    "metadata": {
                        "keywords": ["refund", "returns", "capture"],
                        "topic": "refund",
                    },
                    "score": 0.92,
                },
                {
                    "chunk_id": "cd2",
                    "account_id": "cedar",
                    "content": "Settlement files are published at 04:30 UTC each day.",
                    "metadata": {
                        "keywords": ["settlement", "files", "daily"],
                        "topic": "settlement",
                    },
                    "score": 0.8,
                },
            ],
        }
        self.fail_next = False
        self.delay_seconds = 0.05

    async def search(self, query: str, tenant_id: str, top_k: int) -> list[Evidence]:
        await asyncio.sleep(self.delay_seconds)

        if self.fail_next:
            self.fail_next = False
            raise TimeoutError("simulated vector dependency timeout")

        query_terms = _terms(query)
        if not query_terms:
            return []

        ranked: list[Evidence] = []
        for item in self.docs.get(tenant_id, []):
            doc = Evidence.model_validate(item)
            matched = len(query_terms & _doc_terms(doc))
            if matched:
                relevance = matched / len(query_terms)
                ranked.append(doc.model_copy(update={"score": relevance}))

        ranked.sort(key=lambda d: (-d.score, d.chunk_id))
        return ranked[:top_k]


vector_client = VectorStoreClient()
