# 006: Adopt the Offline (batch) STT Backend Contract

**Status**: Implemented (interface-only)

## Problem

asr-server is standardizing on a single offline-STT adapter so every batch
backend (the new Indic/omniASR services *and* Whisper) speaks one interface.
WhisperX today exposes `/asr` + `/v1/audio/*` with their own request/response
shapes. To let asr-server use one adapter, WhisperX must also speak the
contract in `docs/input/offline-backend-contract.md`.

No consumer calls WhisperX directly (all traffic goes through asr-server), so
this is **purely additive** — existing endpoints are untouched.

## Solution (interface only — no transcription logic changed)

A shared helper module `app/contract.py` plus thin endpoints in both serve
modes (mirroring how `/diarize` was added):

- **`POST /transcribe`** — multipart `audio` + JSON `config`
  (`language`, `task`, `word_timestamps`, `decoding{}`). Wraps the existing
  `run_pipeline` with `should_diarize=False`. The `decoding` bag maps 1:1 onto
  `TranscribeParams`; `initial_prompt`/`hotwords` go to the top-level args.
  Unknown `decoding` keys are ignored. Validated via `TranscribeParams.validate()`.
- **`GET /capabilities`** — advertises model, 100 languages
  (`whisperx.utils.LANGUAGES`), `task: [transcribe, translate]`, and the full
  parameter set with scopes/ranges/defaults.
- **Response conformance** — `{language, duration, text, segments:[{start,end,text,
  words:[{word,start,end,confidence}]}]}`. WhisperX's per-word alignment `score`
  is surfaced as `confidence`; absent fields (unalignable words) are omitted.
- **Errors** — non-2xx `{"error":"...","code":"..."}` (distinct from the
  OpenAI-compat error shape).

Endpoints added to `app/main.py` (simple mode) and `app/serve_app.py` (Ray
ingress, replicate + split).

## Already conformant (no change needed)

- **Audio formats** — `whisperx.load_audio` (ffmpeg) accepts WAV/FLAC/MP3/OGG/M4A.
- **Long files** — handled by VAD chunking + `BATCH_SIZE`.
- **Silence** — returns empty segments (plan 002's VAD fix), satisfying
  acceptance test #3 (no hallucinated text).

## Open items flagged to the team (NOT code changes)

1. **`/health` body shape.** Contract wants `200 {"status":"ok"}`; WhisperX
   returns `{"status":"healthy", ...}`. The adoption brief says keep `/health`
   unchanged, so this was left as-is — needs a decision on whether asr-server's
   adapter checks `status == "ok"`.
2. **Acceptance #5 (RTFx + peak VRAM report)** is a benchmarking deliverable,
   not an interface change — tracked with the ongoing BATCH_SIZE perf work.

## Files

- `app/contract.py` — new shared helpers (config parsing, response formatting,
  capabilities, contract errors)
- `app/main.py` — `POST /transcribe`, `GET /capabilities` (simple mode)
- `app/serve_app.py` — same two endpoints on the Ray Serve ingress
- `CLAUDE.md` — endpoints list + module structure

## Verification

```bash
# Capabilities
curl -s http://localhost:9000/capabilities | jq '.model, (.languages|length), .task'

# Contract transcription (word timestamps + a decoding knob)
curl -s -F "audio=@test.wav" \
  -F 'config={"language":"en","task":"transcribe","word_timestamps":true,"decoding":{"beam_size":5}}' \
  http://localhost:9000/transcribe | jq '{language, duration, nseg:(.segments|length)}'

# Out-of-range knob -> contract-shaped 400
curl -s -F "audio=@test.wav" -F 'config={"decoding":{"vad_chunk_size":120}}' \
  http://localhost:9000/transcribe   # -> {"error":"...","code":"invalid_decoding"}

# Silence -> empty transcript, no hallucination
curl -s -F "audio=@silence.wav" -F 'config={}' http://localhost:9000/transcribe
```
