import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = BASE_DIR / "uploads"
BACKEND_PUBLIC_BASE_URL = os.getenv("BACKEND_PUBLIC_BASE_URL", "http://127.0.0.1:8080")


def ensure_local_dir(subdir: str) -> Path:
    path = LOCAL_STORAGE_DIR / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_bytes_local(subdir: str, filename: str, data: bytes) -> str:
    folder = ensure_local_dir(subdir)
    file_path = folder / filename
    file_path.write_bytes(data)
    return str(file_path)


def save_text_local(subdir: str, filename: str, text: str) -> str:
    folder = ensure_local_dir(subdir)
    file_path = folder / filename
    file_path.write_text(text, encoding="utf-8")
    return str(file_path)


def file_url_for_local_path(file_path: str) -> str:
    """
    Convert backend local storage files to browser-usable URLs.
    Falls back to the original path when it is outside LOCAL_STORAGE_DIR.
    """
    p = Path(file_path).resolve()
    try:
        rel = p.relative_to(LOCAL_STORAGE_DIR.resolve())
    except ValueError:
        return file_path
    return f"{BACKEND_PUBLIC_BASE_URL}/storage/{rel.as_posix()}"


def save_upload(job_id: str, filename: str, data: bytes) -> str:
    """
    Save uploaded file locally under uploads/<job_id>/.
    Returns the absolute file path.
    """
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    file_path = job_dir / filename
    file_path.write_bytes(data)
    return str(file_path)
