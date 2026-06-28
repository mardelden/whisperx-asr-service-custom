"""
Offline (batch) STT backend contract — shared helpers.

Implements the interface described in docs/input/offline-backend-contract.md so
asr-server can talk to WhisperX with the same adapter it uses for the other
offline STT backends. This is purely an interface layer over the existing
WhisperX pipeline — no transcription logic lives here.

  POST /transcribe    multipart (audio + JSON config) -> contract response
  GET  /capabilities  advertise params + languages + tasks

These helpers are consumed by both serve modes:
  - app/main.py        (simple mode endpoints)
  - app/serve_app.py   (Ray Serve ingress endpoints)
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from fastapi.responses import JSONResponse

from whisperx.utils import LANGUAGES

from app.pipeline import TranscribeParams, sanitize_float_values

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Decoding-bag keys that map 1:1 onto TranscribeParams fields. Unknown keys are
# ignored per the contract. `initial_prompt` and `hotwords` are handled
# separately because they are top-level run_pipeline args, not TranscribeParams.
_DECODING_PARAM_FIELDS = [
    "beam_size", "best_of", "patience", "length_penalty",
    "repetition_penalty", "no_repeat_ngram_size",
    "compression_ratio_threshold", "log_prob_threshold",
    "no_speech_threshold", "condition_on_previous_text",
    "prompt_reset_on_temperature", "suppress_blank",
    "without_timestamps", "max_initial_timestamp",
    "max_new_tokens", "hallucination_silence_threshold",
    "prefix", "prepend_punctuations", "append_punctuations",
    "suppress_numerals", "vad_onset", "vad_offset", "batch_size",
]


def contract_error(status_code: int, message: str, code: str) -> JSONResponse:
    """Build a contract-shaped error response: {"error": ..., "code": ...}."""
    return JSONResponse(status_code=status_code, content={"error": message, "code": code})


def _to_param_list(value: Any) -> Optional[str]:
    """Normalize a list-or-string knob into the comma-separated string that
    TranscribeParams expects (used for `temperatures` and `clip_timestamps`)."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _to_hotwords(value: Any) -> Optional[str]:
    """faster-whisper hotwords is a single string; accept list or string."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def parse_config(config_str: str) -> Dict[str, Any]:
    """Parse the JSON `config` part. Raises ValueError on malformed JSON."""
    if not config_str or not config_str.strip():
        return {}
    try:
        parsed = json.loads(config_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid config JSON: {e}")
    if not isinstance(parsed, dict):
        raise ValueError("config must be a JSON object")
    return parsed


def build_transcribe_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Map a contract `config` object onto run_pipeline kwargs.

    Returns a dict with: language, task, word_timestamps, initial_prompt,
    hotwords, params (TranscribeParams). Unknown `decoding` keys are ignored.
    """
    language = config.get("language")
    if language in ("", "auto", None):
        language = None  # trigger auto-detection

    task = config.get("task", "transcribe")
    word_timestamps = bool(config.get("word_timestamps", True))

    decoding = config.get("decoding") or {}
    if not isinstance(decoding, dict):
        raise ValueError("config.decoding must be a JSON object")

    initial_prompt = decoding.get("initial_prompt")
    hotwords = _to_hotwords(decoding.get("hotwords"))

    params = TranscribeParams()
    for field_name in _DECODING_PARAM_FIELDS:
        if field_name in decoding and decoding[field_name] is not None:
            setattr(params, field_name, decoding[field_name])

    # Aliases / type normalization
    if "vad_chunk_size" in decoding and decoding["vad_chunk_size"] is not None:
        params.chunk_size = decoding["vad_chunk_size"]
    elif "chunk_size" in decoding and decoding["chunk_size"] is not None:
        params.chunk_size = decoding["chunk_size"]

    if decoding.get("temperatures") is not None:
        params.temperatures = _to_param_list(decoding["temperatures"])
    if decoding.get("clip_timestamps") is not None:
        params.clip_timestamps = _to_param_list(decoding["clip_timestamps"])

    return {
        "language": language,
        "task": task,
        "word_timestamps": word_timestamps,
        "initial_prompt": initial_prompt,
        "hotwords": hotwords,
        "params": params,
    }


def format_response(result: dict, duration: float, word_timestamps: bool) -> dict:
    """Conform a WhisperX pipeline result to the contract response shape:

        {language, duration, text, segments:[{start,end,text,words:[...]}]}

    Each word: {word, start, end, confidence}. WhisperX's per-word alignment
    `score` is surfaced as `confidence`. Missing fields (unalignable words) are
    omitted, matching the contract.
    """
    segments_out: List[Dict[str, Any]] = []
    text_parts: List[str] = []

    for seg in result.get("segments", []):
        seg_text = (seg.get("text") or "").strip()
        text_parts.append(seg_text)
        seg_out: Dict[str, Any] = {
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": seg_text,
        }
        if word_timestamps and "words" in seg:
            words_out = []
            for w in seg["words"]:
                word_out: Dict[str, Any] = {"word": w.get("word", "")}
                if "start" in w:
                    word_out["start"] = w["start"]
                if "end" in w:
                    word_out["end"] = w["end"]
                if "score" in w:  # WhisperX alignment score -> contract confidence
                    word_out["confidence"] = w["score"]
                words_out.append(word_out)
            seg_out["words"] = words_out
        segments_out.append(seg_out)

    response = {
        "language": result.get("language", "en"),
        "duration": round(duration, 3),
        "text": " ".join(p for p in text_parts if p).strip(),
        "segments": segments_out,
    }
    return sanitize_float_values(response)


def build_capabilities(model_name: str, batch_size: int) -> dict:
    """Advertise WhisperX's supported parameters, languages, and tasks."""
    return {
        "model": model_name,
        "languages": sorted(LANGUAGES.keys()),
        "task": ["transcribe", "translate"],
        "parameters": [
            # Fixed (top-level config) params
            {"key": "language", "scope": "fixed", "type": "string", "default": "auto"},
            {"key": "task", "scope": "fixed", "type": "enum",
             "values": ["transcribe", "translate"], "default": "transcribe"},
            {"key": "word_timestamps", "scope": "fixed", "type": "bool", "default": True},
            # Beam search
            {"key": "beam_size", "scope": "decoding", "type": "int", "range": [1, 16], "default": 5},
            {"key": "best_of", "scope": "decoding", "type": "int", "range": [1, 16], "default": 5},
            {"key": "patience", "scope": "decoding", "type": "float", "default": 1.0},
            {"key": "length_penalty", "scope": "decoding", "type": "float", "default": 1.0},
            # Fallback temperature schedule
            {"key": "temperatures", "scope": "decoding", "type": "float[]",
             "default": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]},
            # Vocabulary / context biasing
            {"key": "initial_prompt", "scope": "decoding", "type": "string", "default": None},
            {"key": "prefix", "scope": "decoding", "type": "string", "default": None},
            {"key": "hotwords", "scope": "decoding", "type": "string", "default": None},
            # Hallucination guards
            {"key": "compression_ratio_threshold", "scope": "decoding", "type": "float", "default": 2.4},
            {"key": "log_prob_threshold", "scope": "decoding", "type": "float", "default": -1.0},
            {"key": "no_speech_threshold", "scope": "decoding", "type": "float", "default": 0.6},
            {"key": "hallucination_silence_threshold", "scope": "decoding", "type": "float", "default": None},
            # Decoding behavior
            {"key": "condition_on_previous_text", "scope": "decoding", "type": "bool", "default": False},
            {"key": "suppress_blank", "scope": "decoding", "type": "bool", "default": True},
            {"key": "suppress_numerals", "scope": "decoding", "type": "bool", "default": False},
            # Anti-loop
            {"key": "repetition_penalty", "scope": "decoding", "type": "float", "default": 1.0},
            {"key": "no_repeat_ngram_size", "scope": "decoding", "type": "int", "default": 0},
            # WhisperX VAD / segmentation
            {"key": "vad_onset", "scope": "decoding", "type": "float", "range": [0.0, 1.0], "default": 0.5},
            {"key": "vad_offset", "scope": "decoding", "type": "float", "range": [0.0, 1.0], "default": 0.363},
            {"key": "vad_chunk_size", "scope": "decoding", "type": "int", "range": [1, 30], "default": 30},
            {"key": "batch_size", "scope": "decoding", "type": "int", "default": batch_size},
        ],
    }


async def read_config_part(form) -> str:
    """Extract the `config` part from a parsed multipart form.

    Accepts it either as a plain text field (most clients) or as a JSON file
    part (content-type application/json). Returns the raw JSON string.
    """
    raw = form.get("config")
    if raw is None:
        return "{}"
    if hasattr(raw, "read"):  # starlette UploadFile (JSON file part)
        return (await raw.read()).decode("utf-8")
    return raw
