import asyncio
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable
from uuid import uuid4

from .models import AsyncJobAccepted, AsyncJobStatusResponse, TranscribeResult

JobRunner = Callable[[], Awaitable[TranscribeResult]]

_JOBS: dict[str, AsyncJobStatusResponse] = {}
_LOCK = asyncio.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _set_job(job: AsyncJobStatusResponse) -> None:
    async with _LOCK:
        _JOBS[job.job_id] = job


async def _update_job(
    job_id: str,
    *,
    status: str | None = None,
    result: TranscribeResult | None = None,
    error: str | None = None,
) -> AsyncJobStatusResponse | None:
    async with _LOCK:
        existing = _JOBS.get(job_id)
        if not existing:
            return None
        updated = existing.model_copy(
            update={
                "status": status or existing.status,
                "updated_at": _now_iso(),
                "result": result if result is not None else existing.result,
                "error": error if error is not None else existing.error,
            }
        )
        _JOBS[job_id] = updated
        return updated


async def purge_expired_jobs(retention_seconds: int) -> None:
    if retention_seconds <= 0:
        return
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=retention_seconds)
    async with _LOCK:
        expired = [
            job_id
            for job_id, job in _JOBS.items()
            if job.status in {"complete", "failed"}
            and datetime.fromisoformat(job.updated_at) < cutoff
        ]
        for job_id in expired:
            _JOBS.pop(job_id, None)


async def submit_job(
    filename: str,
    runner: JobRunner,
    retention_seconds: int,
) -> AsyncJobAccepted:
    await purge_expired_jobs(retention_seconds)

    now = _now_iso()
    job = AsyncJobStatusResponse(
        job_id=uuid4().hex,
        status="queued",
        created_at=now,
        updated_at=now,
        filename=filename,
    )
    job_id = job.job_id
    await _set_job(job)

    async def _execute() -> None:
        await _update_job(job_id, status="running")
        try:
            result = await runner()
            await _update_job(job_id, status="complete", result=result)
        except Exception as exc:
            await _update_job(job_id, status="failed", error=str(exc))

    asyncio.create_task(_execute())
    return AsyncJobAccepted(job_id=job_id, status="queued")


async def get_job(job_id: str, retention_seconds: int) -> AsyncJobStatusResponse | None:
    await purge_expired_jobs(retention_seconds)
    async with _LOCK:
        job = _JOBS.get(job_id)
        return job.model_copy() if job else None
