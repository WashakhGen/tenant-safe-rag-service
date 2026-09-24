import logging

from fastapi import FastAPI, Header, HTTPException, status

from .llm import model_client
from .schemas import AnswerRequest, AnswerResponse
from .utils import VectorUnavailableError, search_with_retry

logger = logging.getLogger(__name__)

app = FastAPI(title="Dvxel AI Knowledge Service")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/answer", response_model=AnswerResponse)
async def answer(
    payload: AnswerRequest,
    trusted_id: str | None = Header(default=None, alias="X-Account-ID"),
):
    if not trusted_id or not trusted_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="trusted tenant header is required",
        )
    if payload.account_override and payload.account_override != trusted_id:
        logger.warning(
            f"ignoring account_override={payload.account_override[:64]!r}; "
            f"using trusted tenant {trusted_id!r}"
        )

    try:
        results = await search_with_retry(payload.question, trusted_id, payload.top_k)
    except VectorUnavailableError:
        logger.error("vector store unavailable; returning degraded response")
        return AnswerResponse(
            account_id=trusted_id,
            answer="The knowledge base is temporarily unavailable.",
            sources=[],
            degraded=True,
        )

    answer_text = await model_client.answer(payload.question, results)
    return AnswerResponse(
        account_id=trusted_id,
        answer=answer_text,
        sources=results,
        degraded=False,
    )
