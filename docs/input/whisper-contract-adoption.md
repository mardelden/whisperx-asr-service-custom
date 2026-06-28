# WhisperX — adopt the offline backend contract (team handoff)

**Mission:** bring the **existing** WhisperX service (`whisper:9000`) into conformance with
the [offline backend contract](./offline-backend-contract.md), so every STT backend —
the new Indic/omniASR ones *and* Whisper — speaks the **same interface** and asr-server
needs only one offline adapter.

**Why this is low-risk:** **no consumer calls WhisperX directly** — all traffic goes
through asr-server. So this is an **additive** change (add the contract endpoints alongside
the current ones); nothing external breaks. Once you ship it, we update asr-server's
`whisperx_service.py` to call the new `/transcribe` and then the legacy `/asr` path can be
retired on our schedule.

## Current API vs the contract (the delta)

| Capability | Contract expects | WhisperX today | Action |
|---|---|---|---|
| Transcribe endpoint | `POST /transcribe` (multipart: `audio` + JSON `config`) | `POST /asr`, `POST /v1/audio/transcriptions` | **Add `/transcribe`** (can wrap existing pipeline) |
| Health | `GET /health` → `{"status":"ok"}` when model ready | `GET /health` ✓ | ✅ already conforms (confirm body shape) |
| Capabilities | `GET /capabilities` (param + language schema) | — (none) | **Add `/capabilities`** |
| Request config | `config` JSON: `language`, `task`, `word_timestamps`, `decoding{}` | own form/query params | **Accept the contract config**, map `decoding` keys → WhisperX params |
| Response | `{language, duration, text, segments:[{start,end,text,words:[{word,start,end,confidence}]}]}` | `TranscriptionSegment`/`Word` (close, different shape) | **Conform the JSON shape** |
| Unknown params | ignore silently | — | don't error on unknown `decoding` keys |

## What needs to be done (conceptual scope)
- **Add `POST /transcribe`** implementing the contract (audio + `config`), reusing your
  existing WhisperX pipeline under the hood. Map the `decoding` bag → WhisperX/faster-whisper
  args (you already support these internally — see Parameters below).
- **Add `GET /capabilities`** advertising Whisper's parameter set + the 99 languages +
  `task: [transcribe, translate]`.
- **Conform the `/transcribe` response** to the contract schema (segments + word
  timestamps + `language` + `duration`).
- **Keep `/asr`, `/v1/audio/transcriptions`, `/diarize`, `/health`, `/metrics` unchanged**
  for now — additive only, no breakage during cutover.
- Pass the [offline contract acceptance test](./offline-backend-contract.md#acceptance-test-every-offline-backend-must-pass).

## Parameters to expose
WhisperX is the richest backend — advertise the full Whisper decoding set via
`/capabilities` and accept it through the `decoding` bag (these mirror asr-server's
existing `DecoderConfig`/`VadConfig`, so the mapping is 1:1):

| Param | Where | Notes |
|---|---|---|
| `language` | fixed | BCP-47 / auto (99 langs) |
| `task` | fixed | transcribe \| translate |
| `word_timestamps` | fixed | WhisperX alignment |
| `beam_size`, `best_of`, `patience`, `length_penalty` | decoding | beam search |
| `temperatures` | decoding | fallback temperature schedule |
| `initial_prompt`, `prefix` | decoding | vocabulary/context biasing |
| `compression_ratio_threshold`, `log_prob_threshold`, `no_speech_threshold`, `hallucination_silence_threshold` | decoding | hallucination guards |
| `condition_on_previous_text`, `suppress_blank`, `suppress_numerals` | decoding | decoding behavior |
| `repetition_penalty`, `no_repeat_ngram_size` | decoding | anti-loop |
| `vad_onset`, `vad_offset`, `vad_chunk_size`, `batch_size` | decoding | WhisperX VAD/segmentation |

(See `proto/asr/v1/speech.proto` `DecoderConfig`/`VadConfig` for the exact set + defaults
asr-server already passes today — the contract just relocates them into the `decoding` bag.)

## Out of scope
Speaker-ID, TSE, the client API — asr-server owns those. (Your existing `/diarize` stays as
a separate endpoint; the contract `/transcribe` is transcription-only.)

## Acceptance
- `POST /transcribe` with a clip → contract-shaped response with segments + word timestamps.
- `GET /capabilities` lists the parameters + languages above.
- Existing `/asr` + `/v1/*` still work (no regression).
- Then we flip asr-server's `whisperx_service.py` to `/transcribe`.
