"""Firebase client singletons for Firestore and Cloud Storage."""

import os
from google.cloud import firestore, storage

_db = None
_bucket = None


def get_db() -> firestore.Client:
    global _db
    if _db is None:
        _db = firestore.Client()
    return _db


def get_bucket() -> storage.Bucket:
    global _bucket
    if _bucket is None:
        bucket_name = os.environ.get("STORAGE_BUCKET")
        if not bucket_name:
            # Default Firebase Storage bucket: <project-id>.firebasestorage.app
            project = os.environ.get("GCLOUD_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
            bucket_name = f"{project}.firebasestorage.app"
        client = storage.Client()
        _bucket = client.bucket(bucket_name)
    return _bucket


def next_id(collection_name: str) -> int:
    """Auto-increment integer ID using a Firestore counter document."""
    db = get_db()
    counter_ref = db.collection("_counters").document(collection_name)

    @firestore.transactional
    def increment(transaction):
        snapshot = counter_ref.get(transaction=transaction)
        current = snapshot.get("value") if snapshot.exists else 0
        new_val = current + 1
        transaction.set(counter_ref, {"value": new_val})
        return new_val

    return increment(db.transaction())
