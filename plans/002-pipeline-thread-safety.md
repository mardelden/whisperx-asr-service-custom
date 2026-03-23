# 002: Pipeline Thread-Safety for Concurrent GPU Requests

**Status**: Implemented (PR #1, merged)

## Problem

With `GPU_CONCURRENCY=20`, multiple requests enter the pipeline concurrently via `ThreadPoolExecutor`. The WhisperX model singleton is not thread-safe — it mutates shared state (`hotwords`, `initial_prompt`, `self.tokenizer`, `self.options`) during `transcribe()`.

## Solution

Per-stage locks + conditional GPU memory clearing:

- `_transcribe_lock`: serializes transcription (required — model not thread-safe)
- `_diarize_lock`: serializes diarization (precautionary — pyannote unverified)
- `align()`: no lock (confirmed stateless and thread-safe)
- In-flight GPU counter: `torch.cuda.empty_cache()` only runs when no other request is on the GPU

This gives **pipeline parallelism** — multiple requests can be in different stages simultaneously — but transcription itself is serialized.

```
Request A: [transcribe] ──────────>  [align] ──────────>  [diarize] ──────────>
Request B:     (waiting on lock)     [align] ──────────>     (waiting on lock)
```

## Limitation

Transcription is the bottleneck — only one request transcribes at a time. The model singleton cannot safely handle concurrent calls.
