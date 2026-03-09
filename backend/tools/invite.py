"""Firestore-backed invite code validation."""

import os
from datetime import datetime, timezone

_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")


async def consume_invite_code(code: str) -> bool:
    """
    Atomically increments use_count if the code is valid and not exhausted/expired.
    Returns True if consumed, False if invalid/exhausted/expired.

    If GOOGLE_CLOUD_PROJECT is not set (local dev), always returns True.
    """
    if not _PROJECT:
        return True  # auth bypassed in local dev

    from google.cloud import firestore  # lazy import — not installed in local dev

    db = firestore.AsyncClient(project=_PROJECT)
    ref = db.collection("invite_codes").document(code.strip().lower())

    @firestore.async_transactional
    async def _txn(transaction):
        snap = await ref.get(transaction=transaction)
        if not snap.exists:
            return False

        data = snap.to_dict()
        max_uses: int = data.get("max_uses", 1)
        use_count: int = data.get("use_count", 0)
        expires_at = data.get("expires_at")

        if use_count >= max_uses:
            return False

        if expires_at is not None:
            # expires_at may be a Firestore Timestamp or a datetime
            exp_dt = expires_at if isinstance(expires_at, datetime) else expires_at.replace(tzinfo=timezone.utc)
            if exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=timezone.utc)
            if datetime.now(tz=timezone.utc) > exp_dt:
                return False

        transaction.update(ref, {"use_count": use_count + 1})
        return True

    txn = db.transaction()
    result = await _txn(txn)
    db.close()
    return result
