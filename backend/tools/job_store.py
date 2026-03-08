"""
In-memory job store. Swap for Firestore in production.

Job shape:
{
    "job_id":    str,
    "status":    "processing" | "done" | "error",
    "step":      str,        # current pipeline step
    "video_url": str | None, # signed URL set by StitcherAgent
    "manifest":  dict | None,# set on completion for LiveAgent context
    "error":     str | None,
}
"""

import threading

_store: dict[str, dict] = {}
_lock = threading.Lock()


def create_job(job_id: str) -> dict:
    job = {
        "job_id": job_id,
        "status": "processing",
        "step": "queued",
        "video_url": None,
        "manifest": None,
        "knowledge_base": None,
        "error": None,
    }
    with _lock:
        _store[job_id] = job
    return job


def get_job(job_id: str) -> dict | None:
    with _lock:
        return _store.get(job_id)


def update_job(job_id: str, **kwargs):
    with _lock:
        if job_id in _store:
            _store[job_id].update(kwargs)
