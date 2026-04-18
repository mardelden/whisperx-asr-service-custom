# 005: POST /diarize — Standalone Speaker Diarization Endpoint

**Status**: Implemented

## Problem

The upstream asr-server needed speaker diarization for `IdentifySpeakers` and `EnrollFromDiarization` RPCs but had to call `/asr` (full Whisper + alignment + diarization), then discard the text. Wasteful — Whisper transcription is the slowest stage.

## Solution

New `POST /diarize` endpoint that runs pyannote diarization directly, skipping Whisper and alignment. 3-10x faster than `/asr`.

### Request

```
POST /diarize
Content-Type: multipart/form-data
Parts: audio_file (bytes)
Query params: num_speakers (int), min_speakers (int), max_speakers (int)
```

### Response

```json
{
  "segments": [
    {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"},
    {"start": 2.5, "end": 5.1, "speaker": "SPEAKER_01"}
  ],
  "num_speakers": 2,
  "audio_duration": 8.3
}
```

## Design Decisions

- **No VAD onset/offset params**: Pyannote's diarization uses its own learned segmentation model, not WhisperX's VAD thresholds. These params don't apply.
- **Separate `diarize_only()` function**: The existing `diarize()` always calls `assign_word_speakers()` which requires transcript segments. `diarize_only()` is a clean standalone path.
- **GPU queue**: Uses `run_in_queue()` for backpressure, same as `/asr`.
- **HF_TOKEN required**: Returns 400 if missing (not a silent degradation like `/asr`).

## Files Modified

- `app/pipeline.py` — Added `diarize_only()` function
- `app/main.py` — Added `POST /diarize` endpoint
- `app/serve_app.py` — Added `/diarize` to Ray Serve ingress
- `app/serve_deployments.py` — Added `diarize_only` method to `FullPipelineDeployment` and `DiarizeDeployment`
- `CLAUDE.md` — Updated endpoints list
