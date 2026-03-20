"""
Streaming file upload utility.

Reads uploaded files in chunks to avoid loading the entire file into memory.
"""

import os
import tempfile
import logging
from typing import Optional
from pathlib import Path

from fastapi import UploadFile

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "1000"))
UPLOAD_CHUNK_SIZE_BYTES = int(os.getenv("UPLOAD_CHUNK_SIZE_BYTES", str(8 * 1024 * 1024)))  # 8MB default


class FileTooLargeError(Exception):
    """Raised when an uploaded file exceeds MAX_FILE_SIZE_MB."""

    def __init__(self, file_size_mb: float, max_size_mb: int):
        self.file_size_mb = file_size_mb
        self.max_size_mb = max_size_mb
        super().__init__(
            f"File too large ({file_size_mb:.1f}MB). Maximum allowed: {max_size_mb}MB. "
            f"Large files may cause out-of-memory errors."
        )


async def save_upload_to_tempfile(
    upload_file: UploadFile,
    content_length: Optional[int] = None,
) -> tuple[str, float]:
    """Stream an UploadFile to a temporary file on disk, checking size limits.

    Args:
        upload_file: The FastAPI UploadFile to save.
        content_length: Optional Content-Length header value for early rejection.

    Returns:
        Tuple of (temp_file_path, file_size_mb).

    Raises:
        FileTooLargeError: If the file exceeds MAX_FILE_SIZE_MB.
    """
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

    # Early rejection via Content-Length header (advisory)
    if content_length is not None and content_length > max_bytes:
        raise FileTooLargeError(content_length / (1024 * 1024), MAX_FILE_SIZE_MB)

    suffix = Path(upload_file.filename).suffix if upload_file.filename else ".wav"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            total_bytes = 0

            while True:
                chunk = await upload_file.read(UPLOAD_CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise FileTooLargeError(total_bytes / (1024 * 1024), MAX_FILE_SIZE_MB)
                temp_file.write(chunk)

        file_size_mb = total_bytes / (1024 * 1024)

        if file_size_mb > 100:
            logger.warning(f"Processing large file ({file_size_mb:.1f}MB) - may consume significant VRAM")

        return temp_path, file_size_mb

    except FileTooLargeError:
        # Clean up partial temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise
    except OSError:
        # Disk full or other I/O error — clean up partial file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise
