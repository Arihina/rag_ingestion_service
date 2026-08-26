from __future__ import annotations


"""Реестр задач в памяти процесса.
Временное решение
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class Job:
    job_id: str
    filename: str
    rag_id: str
    document_id: str
    status: JobStatus = JobStatus.PENDING
    chunks_loaded: int = 0
    error: str | None = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None


class JobRegistry:
    def __init__(self, max_entries: int = 1000) -> None:
        self._jobs: dict[str, Job] = {}
        self._max_entries = max_entries

    def create(self, filename: str, rag_id: str, document_id: str) -> Job:
        job = Job(
            job_id=uuid.uuid4().hex,
            filename=filename,
            rag_id=rag_id,
            document_id=document_id,
        )
        self._jobs[job.job_id] = job
        self._evict()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def pending_count(self) -> int:
        return sum(
            1
            for j in self._jobs.values()
            if j.status in (JobStatus.PENDING, JobStatus.RUNNING)
        )

    def _evict(self) -> None:
        if len(self._jobs) <= self._max_entries:
            return
        finished = sorted(
            (j for j in self._jobs.values() if j.finished_at),
            key=lambda j: j.finished_at,  # type: ignore[arg-type]
        )
        for job in finished[: len(self._jobs) - self._max_entries]:
            self._jobs.pop(job.job_id, None)


class JobResponse(BaseModel):
    job_id: str
    filename: str
    rag_id: str
    document_id: str
    status: JobStatus
    chunks_loaded: int = 0
    error: str | None = None

    @classmethod
    def of(cls, job: Job) -> "JobResponse":
        return cls(
            job_id=job.job_id,
            filename=job.filename,
            rag_id=job.rag_id,
            document_id=job.document_id,
            status=job.status,
            chunks_loaded=job.chunks_loaded,
            error=job.error,
        )
