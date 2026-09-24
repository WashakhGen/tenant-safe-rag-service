import asyncio
import logging

from .config import settings
from .schemas import Evidence
from .vector import vector_client

logger = logging.getLogger(__name__)


class VectorUnavailableError(Exception):
    """The vector store failed after all retries."""


async def search_with_retry(
    question: str, tenant_id: str, top_k: int
) -> list[Evidence]:
    # 1 initial try + max_retries retries
    for attempt in range(settings.max_retries + 1):
        try:
            return await asyncio.wait_for(
                vector_client.search(question, tenant_id, top_k),
                timeout=settings.vector_timeout_seconds,
            )
        except Exception:
            logger.warning(
                "vector search failed (attempt %d/%d)",
                attempt + 1,
                settings.max_retries + 1,
                exc_info=True,
            )
    raise VectorUnavailableError
