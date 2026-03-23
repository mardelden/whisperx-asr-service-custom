# 004: Expose All ASR & VAD Options as Per-Request Parameters

**Status**: Implemented

## Problem

Only `initial_prompt` and `hotwords` were configurable per-request. All other WhisperX tuning knobs (beam_size, VAD thresholds, repetition_penalty, etc.) were hardcoded at model load time.

## Solution

`TranscribeParams` dataclass in `app/pipeline.py` holds optional overrides for all WhisperX transcription options. The `transcribe()` function uses snapshot/mutate/restore on the cached model singleton:

1. `copy.copy(whisper_model.options)` → save original state
2. Apply non-None overrides from `TranscribeParams`
3. Call `whisper_model.transcribe()`
4. Restore originals in `finally` block

No model reload needed — `TranscriptionOptions` is a mutable dataclass on the production server (`faster_whisper==1.2.1`).

## Per-Request Parameters (all optional, default = model default)

**ASR Options**: beam_size, best_of, patience, length_penalty, repetition_penalty, no_repeat_ngram_size, temperatures, compression_ratio_threshold, log_prob_threshold, no_speech_threshold, condition_on_previous_text, prompt_reset_on_temperature, suppress_blank, without_timestamps, max_initial_timestamp, suppress_numerals, max_new_tokens, clip_timestamps, hallucination_silence_threshold, prefix, prepend_punctuations, append_punctuations

**VAD Options**: vad_onset, vad_offset, chunk_size

**Other**: batch_size

## Thread Safety Note

With `GPU_CONCURRENCY=1` (current default), only one transcription runs at a time — snapshot/restore is safe. If plan 002's `_transcribe_lock` is applied later, it serializes these mutations too.

## Files Modified

- `app/pipeline.py` — `TranscribeParams` dataclass, updated `transcribe()` and `run_pipeline()`
- `app/main.py` — ~25 new query params on `/asr`, constructs `TranscribeParams`
- `app/openai_compat.py` — `params` passthrough in `_run_transcribe_and_align()` and `process_audio()`
- `app/serve_app.py` — same query params on Ray Serve `/asr`, passes `params` via `.remote()`
- `app/serve_deployments.py` — `params` argument on `FullPipelineDeployment.run()`, `WhisperDeployment.transcribe()`
