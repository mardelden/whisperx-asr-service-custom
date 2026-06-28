# Offline (batch) STT Backend Contract v1

**Audience:** any team deploying an **offline / batch** STT service that `asr-server`
will consume (the Telugu/Indic backends — see the per-backend briefs in this directory).
This is the batch analogue of the [streaming contract](./streaming-backend-contract.md):
audio in → transcript out, no incremental partials.

**Why a shared contract:** it's the same pattern as the existing `whisper` sidecar
(`whisper:9000/asr`) — a stable HTTP API that asr-server adapts to. One contract → one
asr-server offline adapter, regardless of model.

---

## How asr-server uses this

```
            client ──gRPC Recognize──▶ asr-server
                                         │  • speaker identification (Qdrant)   ← above the backend
                                         │  • TSE / voice isolation             ← above the backend
                                         │  • model registry / routing
                                         ▼  HTTP (THIS CONTRACT)
                              your offline STT service (own LXC)
                                   audio file  →  transcript + word timestamps
```

**You own** turning a full audio buffer into a transcript with timestamps. **asr-server
owns** the client API, speaker-ID, TSE, registry. Do **not** build those.

## Transport

- **HTTP**. `POST /transcribe` (multipart/form-data) — `audio` file part + a JSON `config`
  part. Returns `application/json`.
- Health: `GET /health` → `200 {"status":"ok"}` only when the model is loaded + GPU-ready.
- **Capabilities: `GET /capabilities`** — advertise supported params (see below) so the
  SaaS can expose them dynamically.

## Audio

- Accept common containers (WAV/FLAC/MP3/OGG/M4A) — detect from header — OR at minimum
  **16 kHz mono PCM16 WAV**. asr-server can pre-transcode, but accepting common formats is
  preferred. Long files (≥ tens of minutes) must be supported (chunk internally if needed).

## Request

```
POST /transcribe   (multipart/form-data)
  audio:  <binary>           # the full clip
  config: {                  # JSON
    "language": "te",        # BCP-47; "" / "auto" = detect if supported
    "task": "transcribe",    # "transcribe" | "translate" (if supported)
    "word_timestamps": true,
    "decoding": {            # OPTIONAL backend-specific knobs; ignore unknown keys
      // e.g. "decoder": "rnnt", "beam_size": 4, "hotwords": ["Vulcandom"]
    }
  }
```

## Response

```json
{
  "language": "te",
  "duration": 73.4,
  "text": "full transcript ...",
  "segments": [
    {
      "start": 0.0, "end": 4.2,
      "text": "segment text",
      "words": [
        { "word": "…", "start": 0.0, "end": 0.3, "confidence": 0.94 }
      ]
    }
  ]
}
```

- `words` may be omitted if `word_timestamps:false`.
- Errors: non-2xx with `{"error":"...","code":"..."}`.

These map onto `asr-server`'s `Recognize` → `RecognizeResponse`
(`SpeechRecognitionResult` + `WordInfo`), exactly like the WhisperX backend today.

## Parameter flexibility (`decoding` + `/capabilities`)

Same design as the streaming contract: model-specific knobs travel in the free-form
`decoding` object (backend **ignores unknown keys**), and `GET /capabilities` advertises
the supported set so asr-server / the SaaS can expose them without hard-coding:

```json
{
  "model": "indic-conformer-600m",
  "languages": ["te","hi","ta","..."],   // or "auto"
  "task": ["transcribe"],
  "parameters": [
    { "key": "decoder", "scope": "decoding", "type": "enum", "values": ["ctc","rnnt"], "default": "rnnt" },
    { "key": "beam_size", "scope": "decoding", "type": "int", "range": [1,16], "default": 1 },
    { "key": "hotwords", "scope": "decoding", "type": "string[]", "default": [] }
  ]
}
```

The exact parameter list per backend is in each brief's **"Parameters to expose"** section.

## Container & infra

- Separate **GPU-passthrough LXC** on the GPU host (shares the GPU; coordinate VRAM).
- Dedicated port (suggested in each brief); `/health` for readiness; Ansible-deployed.

## Acceptance test (every offline backend must pass)

1. `GET /health` → 200 only after the model is loaded.
2. `POST /transcribe` a **Telugu** clip with `word_timestamps:true` → returns a correct
   transcript with segment + word timestamps and the detected/echoed language.
3. `POST /transcribe` 5 s of silence → empty/near-empty transcript, no hallucinated text.
4. `GET /capabilities` lists the supported parameters.
5. Report real-time factor (RTFx) and peak VRAM on our hardware.
