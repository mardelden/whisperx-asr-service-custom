# 003: Cross-Request Batching with Priority Scheduling

**Status**: Future / Exploration

## Motivation

Plan 002 serializes transcription via a lock — only one request transcribes at a time. This leaves GPU utilization suboptimal when batches aren't full (e.g., a short audio clip uses 1 of 16 batch slots).

The goal is vLLM-style continuous batching for Whisper: same VRAM, higher throughput, and priority scheduling so small requests don't wait behind large files.

## How WhisperX Batching Works Today

1. VAD (Voice Activity Detection) splits audio into ~30s segments
2. Segments are grouped into batches of `BATCH_SIZE` (default 16)
3. Each batch is one forward pass through the model
4. All batches for one file run sequentially inside `transcribe()`

A 10-minute file produces ~20 segments → 2 batches. A 5-second voice clip → 1 segment.

## Proposed Design

### Priority Batch Scheduler

Decouple VAD from inference. Run VAD upfront for all requests, then feed segments into a shared priority queue:

```
Large file:  [batch 1: segs 1-16] → [batch 2: segs 17-20 + small request segs] → ...
Small file:                          ↑ inserted here with priority
```

### Components

1. **VAD preprocessing** — run VAD for each incoming request, producing a list of ~30s segments tagged with request ID
2. **Segment priority queue** — collects segments from all pending requests; priority based on request size (small first), arrival time, or explicit priority flag
3. **Batch assembler** — pulls segments from queue, fills batches up to `BATCH_SIZE`, runs inference
4. **Result reassembly** — routes inference output (per-segment transcriptions) back to the correct request, reassembles into final result
5. **Request future/callback** — each request gets an `asyncio.Future` that resolves when all its segments are transcribed

### Priority Strategy Options

- **Shortest-job-first**: requests with fewer segments get priority (small files transcribed faster)
- **Fair interleaving**: round-robin across requests (no starvation)
- **Explicit priority**: API parameter (e.g., `priority=high`) for latency-sensitive requests

### Benefits

- GPU batch slots are always full → higher throughput
- Small requests (e.g., short voice clip) get results fast instead of waiting behind a 1-hour podcast
- Same VRAM footprint — single model instance
- Natural backpressure — queue depth signals load

## Challenges

- **WhisperX coupling**: `transcribe()` internally runs VAD + encode + decode in one call. Need to break this apart or patch upstream.
- **Tokenizer/options mutation**: upstream `FasterWhisperPipeline.transcribe()` mutates `self.tokenizer` and `self.options` — even with batching, the model call itself must be single-threaded unless patched.
- **Language detection**: WhisperX detects language from the first 30s. Cross-request batching may mix languages — need per-segment language handling or require `language` parameter.
- **State reassembly**: segments from different requests interleaved in a batch need correct routing back. Bookkeeping complexity.
- **Error isolation**: one bad segment shouldn't fail other requests in the same batch.

## Alternatives Considered

- **Multiple model instances on same GPU**: load model N times, each in its own thread. Simple but multiplies VRAM (large-v3 ~10GB each).
- **CTranslate2 `inter_threads`**: faster-whisper's backend supports threading, but WhisperX wrapper mutates state on top. Would need upstream patches.
- **Ray Serve split mode**: already supports `@serve.batch` but actors are single-threaded per replica. Scales via replicas, not batch interleaving.

## Next Steps

1. Prototype: extract VAD step from `whisperx.transcribe()` to run independently
2. Measure: profile batch utilization on real workloads to quantify the throughput gain
3. Decide: whether to patch upstream WhisperX or fork the transcribe internals
