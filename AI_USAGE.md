# AI Usage

I used Claude Code as a pair programmer in this take-home.

## Where it helped materially
- **Tests:** drafting the isolation, ranking, failure, concurrency, model, evaluation and grounding tests, including the fixtures that inject faults and count dependency calls.
- **Core logic:** I wrote the term-overlap scorer, the retry helper, and the recall, MRR and grounding functions myself, then used AI to review them for a more compact and efficient implementation. I also used it to look up Python syntax I didn't remember offhand.
- **Review:** I used AI to point out weaknesses in the options I had considered before choosing one.

The design decisions and implementation were mine: header-only tenant identity, ignore-and-log for `account_override`, a degraded 200 response on dependency failure, and a deterministic scorer over a library.

## An output I rejected
The AI raised `rank_bm25` as a ready-made ranking library. Rather than accept it on reputation, I asked for a comparison and it was run against the repo's own data. On the two-document `cedar` tenant every BM25 score was 0, because each query term appears in exactly one of two documents and its IDF is `ln(1) = 0`, so every cedar question would have retrieved nothing. I rejected the library and kept the small overlap scorer, which passes all six benchmark questions. I also declined scikit-learn's stopword list: it adds a roughly 150 MB dependency and contains words such as "bill" and "amount" that can carry meaning in payments questions.

## An output I corrected
Deliberately breaking the code showed that the AI's first whole-phrase grounding test still passed with the whole-phrase rule removed, because a separate sentence check already rejected the answer. The test was rewritten around evidence that says "17 calendar days" with the required fact "7 calendar days", which fails only if whole-phrase matching is broken.
**Review:** I used AI to point out weaknesses in the options I had considered before choosing one.