# WhisperX ASR Service

## Build & Dev Commands

```bash
# Production (prebuilt image)
docker compose up -d

# Dev build (local source, live reload)
docker compose -f docker-compose.dev.yml up --build

# Rebuild after changes
docker compose -f docker-compose.dev.yml up --build --force-recreate

# Stress test
python tests/stress_test.py

# Quick test
curl -F "audio_file=@test.wav" http://localhost:9000/asr
```

## Architecture

FastAPI service wrapping WhisperX with two serve modes:

**Simple mode** (`SERVE_MODE=simple`, default): Single uvicorn process. Async GPU queue serializes pipeline runs via semaphore + thread pool executor. Good for single GPU / low traffic.

**Ray Serve mode** (`SERVE_MODE=ray`): Cross-request batching with `@serve.batch`. Two strategies:
- **Replicate** (`PIPELINE_STRATEGY=replicate`, default): Full 3-stage pipeline per GPU replica.
- **Split** (`PIPELINE_STRATEGY=split`): Each stage as separate Ray deployment with fractional GPU allocation (whisper 0.5, align 0.3, diarize 0.2).

Pipeline: Audio upload → Transcribe (Whisper) → Align (wav2vec2) → Diarize (pyannote) → JSON/SRT/VTT response.

Endpoints: `/asr` (native), `/v1/audio/transcriptions` and `/v1/audio/translations` (OpenAI-compatible), `/health`, `/metrics`.

## Module Structure

```
app/
├── __init__.py            # Package init
├── version.py             # __version__ = "0.3.1"
├── main.py                # Simple mode FastAPI app, /asr endpoint
├── pipeline.py            # Shared 3-stage pipeline: transcribe/align/diarize + model caching
├── queue.py               # Async GPU queue (semaphore + ThreadPoolExecutor)
├── schemas.py             # Pydantic models (OpenAI-compatible responses)
├── openai_compat.py       # /v1/audio/* endpoints, model mapping
├── serve_app.py           # Ray Serve ingress (ASRIngress class)
└── serve_deployments.py   # Ray Serve deployments (FullPipeline, Whisper, Align, Diarize)
```

## Key Patterns

- **Thread-safe model loading**: Double-checked locking in `pipeline.py` — check cache, acquire lock, check again, load. Per-model and per-language caching.
- **Async GPU queue** (`queue.py`): `asyncio.Semaphore` + `ThreadPoolExecutor` keeps event loop responsive while GPU work runs in threads. Configurable via `GPU_CONCURRENCY`.
- **Graceful degradation**: Alignment and diarization catch exceptions and return partial results rather than failing the request.
- **Ray Serve batching**: `@serve.batch` on split-mode deployments collects requests up to `max_batch_size` or `batch_wait_timeout_s` (0.1s default).
- **OpenAI API compatibility**: Model name mapping (whisper-1 → large-v3), response format translation, standard error responses.

## Critical Rules

- **HF_TOKEN required**: Diarization needs a HuggingFace token with access to pyannote models. Set via env var or `.env` file.
- **WhisperX from local source**: Dockerfile installs WhisperX from sibling `whisperX-custom` repo via `additional_contexts` in docker-compose.dev.yml. No remote fork or sed patch needed.
- **Version tag must match `app/version.py`**: Currently `0.3.1`. The entrypoint.sh prints this on startup.
- **GPU memory**: `large-v3` needs ~10GB VRAM. Service clears GPU memory between pipeline stages via `gc.collect()` + `torch.cuda.empty_cache()`.
- **Shared memory**: Ray mode requires `shm_size: 8g` in docker-compose for Ray object store.
- **Base image**: `nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04` with PyTorch 2.3.0 + CUDA 12.1.

## Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SERVE_MODE` | simple | `simple` or `ray` |
| `DEVICE` | cuda | `cuda` or `cpu` |
| `COMPUTE_TYPE` | float16 | `float16`, `float32`, `int8` |
| `BATCH_SIZE` | 16 | Whisper batch size |
| `HF_TOKEN` | (required) | HuggingFace auth token |
| `PRELOAD_MODEL` | large-v3 | Model to load on startup |
| `PIPELINE_STRATEGY` | replicate | `replicate` or `split` (ray mode) |
| `GPU_CONCURRENCY` | 1 | Concurrent GPU runs (simple mode) |
| `NUM_GPU_REPLICAS` | 1 | Pipeline replicas (ray mode) |
