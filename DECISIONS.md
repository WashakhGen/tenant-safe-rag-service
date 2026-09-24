# Decisions

## Trust boundary
The tenant comes only from the `X-Account-ID` header, which the gateway sets after authentication. Request-body fields are untrusted. `account_override` stays in the schema for compatibility but is ignored; a mismatch is logged as a warning (truncated and escaped). Blank or whitespace-only headers return 400. The endpoint passes only the trusted tenant to retrieval, and `search` reads that tenant's documents before ranking or slicing. The shared `request_buffer` was removed, so the handler is stateless. The evaluation also reports `tenant_leaks` as a regression guard.

## Retrieval and evaluation
Retrieval filters to the tenant, scores each chunk by the share of query terms it matches (stopwords removed, plurals stemmed), drops zero-match chunks, sorts by score then `chunk_id` for determinism, and only then applies top-k. I tested `rank_bm25`: on two- and three-document tenants its IDF (Inverse Document Frequency) gave every cedar chunk a score of 0, so I kept the small deterministic scorer.

The model quotes every retrieved chunk once, in rank order, so multi-fact questions work.

Metrics: Recall@k is relevant chunks found divided by relevant chunks; MRR is 1/rank of the first relevant chunk. An answer is grounded only if every required fact appears in both the answer and the retrieved evidence, and every answer sentence appears in the evidence, so invented claims fail. Grounding is reported separately from retrieval, and tests show the two vary independently. The benchmark is unchanged and deterministic. It scores 1.0, so tests use synthetic bad inputs to prove the metrics can fall.

## Failure semantics
Each search runs under `asyncio.wait_for` (0.2 s) with up to `1 + max_retries` attempts (3 by default, about 0.6 s worst case). If all fail, the endpoint returns 200 with `degraded=true`, empty sources and a fixed "temporarily unavailable" answer; the model is not called. It never returns a 500 for a dependency failure and never falls back to other tenants' or cached data. Clients must check `degraded`. The simulated delay uses `asyncio.sleep`, so concurrent requests do not block each other.

## Trade-off
Dropping zero-match chunks and strict sentence-level grounding favour precision over recall. A synonym ("money back") retrieves nothing, and a correct paraphrase fails grounding. I accepted this because irrelevant evidence produces confident wrong answers, and the benchmark needs deterministic metrics. In production I would add embeddings for hybrid search and an entailment or LLM judge for grounding.
