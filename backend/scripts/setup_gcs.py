"""
One-time GCS bucket setup script.

Creates the nevertrtfm bucket with:
- Regional storage in us-central1
- Uniform bucket-level access
- Folders for uploads, clips, final videos, and shared assets

Usage:
  cd backend
  uv run python scripts/setup_gcs.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

from google.cloud import storage

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
BUCKET_NAME = os.getenv("GCS_BUCKET", "nevertrtfm")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

FOLDERS = [
    "uploads/",
    "clips/",
    "final/",
    "shared/avatars/",
]


def main():
    client = storage.Client(project=PROJECT_ID)

    # Create bucket if it doesn't exist
    bucket = client.bucket(BUCKET_NAME)
    if bucket.exists():
        print(f"Bucket gs://{BUCKET_NAME} already exists.")
    else:
        bucket = client.create_bucket(
            BUCKET_NAME,
            project=PROJECT_ID,
            location=LOCATION,
        )
        # Uniform bucket-level access — simpler IAM, no per-object ACLs
        bucket.iam_configuration.uniform_bucket_level_access_enabled = True
        bucket.patch()
        print(f"Created bucket gs://{BUCKET_NAME} in {LOCATION}")

    # Create placeholder objects to materialise folder structure
    for folder in FOLDERS:
        blob = bucket.blob(folder)
        if not blob.exists():
            blob.upload_from_string(b"", content_type="application/x-directory")
            print(f"  Created folder: gs://{BUCKET_NAME}/{folder}")
        else:
            print(f"  Folder exists:  gs://{BUCKET_NAME}/{folder}")

    print(f"""
Done. Update your .env:
  DEV_MODE=false
  GCS_BUCKET={BUCKET_NAME}
""")


if __name__ == "__main__":
    main()
