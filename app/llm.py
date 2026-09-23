class DeterministicModel:
    async def answer(self, question: str, evidence: list):
        if not evidence:
            return (
                "I do not have enough evidence to answer that from the knowledge base."
            )
        return evidence[0].content


model_client = DeterministicModel()
