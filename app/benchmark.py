BENCHMARK = [
    {
        "id": "b1",
        "tenant": "atlas",
        "question": "How long are refunds allowed after capture?",
        "relevant_chunk_ids": ["at1"],
        "required_facts": ["5 calendar days", "capture"],
    },
    {
        "id": "b2",
        "tenant": "atlas",
        "question": "When are settlement files published?",
        "relevant_chunk_ids": ["at2"],
        "required_facts": ["01:30 UTC", "each day"],
    },
    {
        "id": "b3",
        "tenant": "atlas",
        "question": "Where is chargeback evidence uploaded?",
        "relevant_chunk_ids": ["at3"],
        "required_facts": ["disputes console"],
    },
    {
        "id": "b4",
        "tenant": "atlas",
        "question": "Give me the refund window and settlement file time.",
        "relevant_chunk_ids": ["at1", "at2"],
        "required_facts": ["5 calendar days", "01:30 UTC"],
    },
    {
        "id": "b5",
        "tenant": "atlas",
        "question": "What is the returns refund policy?",
        "relevant_chunk_ids": ["at1"],
        "required_facts": ["5 calendar days"],
    },
    {
        "id": "b6",
        "tenant": "atlas",
        "question": "What is the daily settlement file schedule?",
        "relevant_chunk_ids": ["at2"],
        "required_facts": ["01:30 UTC"],
    },
]
