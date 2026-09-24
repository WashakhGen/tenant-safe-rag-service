NO_EVIDENCE_ANSWER: str = (
    "I do not have enough evidence to answer that from the knowledge base."
)


class DeterministicModel:
    async def answer(self, question: str, evidence: list):
        if not evidence:
            return NO_EVIDENCE_ANSWER

        # quote every retrieved chunk once, in ranked order.
        parts: list[str] = []
        for item in evidence:
            text = item.content.strip()
            if text and text not in parts:
                parts.append(text)

        return " ".join(parts) or NO_EVIDENCE_ANSWER


model_client = DeterministicModel()
