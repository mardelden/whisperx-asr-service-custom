# Plan 001: Streaming File Uploads to Avoid Memory Exhaustion

**Status:** Implemented

## Problem

All upload entry points called `await audio_file.read()` loading entire files into RAM before writing to disk. With 1GB+ file limits, each in-flight request consumed 1GB+ of RAM just for the upload step.

## Solution

Created `app/upload.py` with `save_upload_to_tempfile()` that streams uploads to disk in configurable chunks (default 8MB via `UPLOAD_CHUNK_SIZE_BYTES`). Size is validated during streaming, aborting early if `MAX_FILE_SIZE_MB` is exceeded.

## Files Changed

| File | Change |
|------|--------|
| `app/upload.py` | **NEW** — `save_upload_to_tempfile()`, `FileTooLargeError`, centralized `MAX_FILE_SIZE_MB` and `UPLOAD_CHUNK_SIZE_BYTES` |
| `app/main.py` | Replaced in-memory upload with streaming, removed local `MAX_FILE_SIZE_MB`, added `Request` param |
| `app/openai_compat.py` | Replaced in-memory upload with streaming, removed local `MAX_FILE_SIZE_MB`, added `request` param to `process_audio()` |
| `app/serve_app.py` | Replaced both upload blocks with streaming, removed local `MAX_FILE_SIZE_MB`, added `request` params |
| `CLAUDE.md` | Added `upload.py` to module structure, `UPLOAD_CHUNK_SIZE_BYTES` to env var table |
