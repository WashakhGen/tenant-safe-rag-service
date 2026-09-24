import time

from .schemas import Evidence


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
        time.sleep(self.delay_seconds)

        if self.fail_next:
            self.fail_next = False
            raise TimeoutError("simulated vector dependency timeout")

        tenant_docs = self.docs.get(tenant_id, [])
        return [Evidence.model_validate(item) for item in tenant_docs[:top_k]]


vector_client = VectorStoreClient()
