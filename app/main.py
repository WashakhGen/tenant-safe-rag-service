import time

from fastapi import FastAPI, Header, HTTPException

from .llm import model_client
from .schemas import AnswerRequest, AnswerResponse, Evidence
from .vector import vector_client

app = FastAPI(title="Dvxel AI Knowledge Service")
request_buffer: list[Evidence] = []


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/answer", response_model=AnswerResponse)
async def answer(
    payload: AnswerRequest,
    trusted_id: str | None = Header(default=None, alias="X-Account-ID"),
):
    if not trusted_id:
        raise HTTPException(status_code=400, detail="trusted tenant header is required")
    tenant_id = payload.account_override or trusted_id
    started = time.perf_counter()
    degraded = False
    results = await vector_client.search(payload.question, tenant_id, payload.top_k)
    request_buffer.extend(results)
    answer_text = await model_client.answer(
        payload.question, request_buffer[: payload.top_k]
    )
    return AnswerResponse(
        account_id=tenant_id,
        answer=answer_text,
        sources=request_buffer[: payload.top_k],
        degraded=degraded,
    )
