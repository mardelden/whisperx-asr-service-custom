"""
Shared ASR pipeline stage functions.

Extracts the 3-stage WhisperX pipeline (transcribe -> align -> diarize) into
reusable functions consumed by both the legacy FastAPI endpoints and the
Ray Serve deployments.
"""

import os
import gc
import copy
import math
import logging
import dataclasses
import threading
import warnings
from typing import Optional, Dict, Any, List, Tuple

# Suppress pyannote's torchcodec warning -- we decode audio via whisperx.load_audio (ffmpeg),
# not pyannote's built-in decoder, so the missing torchcodec is irrelevant.
warnings.filterwarnings("ignore", message=".*torchcodec.*")

import numpy as np
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (read once at import time, same as before)
# ---------------------------------------------------------------------------
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
HF_TOKEN = os.getenv("HF_TOKEN", None)
CACHE_DIR = os.getenv("CACHE_DIR", "/.cache")
DEFAULT_MODEL = os.getenv("PRELOAD_MODEL", "large-v3")

_model_load_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Model caches
# ---------------------------------------------------------------------------
_whisper_models: Dict[str, Any] = {}
_align_models: Dict[str, Tuple[Any, Any]] = {}
_diarize_pipeline: Optional[DiarizationPipeline] = None


# ---------------------------------------------------------------------------
# Per-request transcription overrides
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class TranscribeParams:
    """Per-request overrides for WhisperX transcription options.

    All fields default to None, meaning 'use the model default'.
    Only non-None values override the cached model's settings.
    """

    # ASR options (TranscriptionOptions fields)
    beam_size: Optional[int] = None
    best_of: Optional[int] = None
    patience: Optional[float] = None
    length_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    no_repeat_ngram_size: Optional[int] = None
    temperatures: Optional[str] = None  # comma-separated → List[float]
    compression_ratio_threshold: Optional[float] = None
    log_prob_threshold: Optional[float] = None
    no_speech_threshold: Optional[float] = None
    condition_on_previous_text: Optional[bool] = None
    prompt_reset_on_temperature: Optional[float] = None
    suppress_blank: Optional[bool] = None
    without_timestamps: Optional[bool] = None
    max_initial_timestamp: Optional[float] = None
    max_new_tokens: Optional[int] = None
    clip_timestamps: Optional[str] = None  # comma-separated → List[float]
    hallucination_silence_threshold: Optional[float] = None
    prefix: Optional[str] = None
    prepend_punctuations: Optional[str] = None
    append_punctuations: Optional[str] = None

    # Instance attribute on FasterWhisperPipeline
    suppress_numerals: Optional[bool] = None

    # VAD params (mutated on whisper_model._vad_params and .vad_model)
    vad_onset: Optional[float] = None
    vad_offset: Optional[float] = None

    # Direct args to whisper_model.transcribe()
    chunk_size: Optional[int] = None
    batch_size: Optional[int] = None

    def has_overrides(self) -> bool:
        return any(v is not None for v in dataclasses.asdict(self).values())

    def validate(self) -> List[str]:
        """Return list of validation error messages. Empty list = valid."""
        errors = []
        # Whisper encoder window is fixed at 30s (3000 mel frames)
        if self.chunk_size is not None and not (1 <= self.chunk_size <= 30):
            errors.append("chunk_size must be between 1 and 30 (Whisper encoder limit is 30s)")
        if self.beam_size is not None and self.beam_size < 1:
            errors.append("beam_size must be >= 1")
        if self.best_of is not None and self.best_of < 1:
            errors.append("best_of must be >= 1")
        if self.patience is not None and self.patience <= 0:
            errors.append("patience must be > 0")
        if self.length_penalty is not None and self.length_penalty <= 0:
            errors.append("length_penalty must be > 0")
        if self.repetition_penalty is not None and self.repetition_penalty < 1.0:
            errors.append("repetition_penalty must be >= 1.0")
        if self.no_repeat_ngram_size is not None and self.no_repeat_ngram_size < 0:
            errors.append("no_repeat_ngram_size must be >= 0")
        if self.no_speech_threshold is not None and not (0.0 <= self.no_speech_threshold <= 1.0):
            errors.append("no_speech_threshold must be between 0.0 and 1.0")
        if self.vad_onset is not None and not (0.0 < self.vad_onset <= 1.0):
            errors.append("vad_onset must be between 0.0 (exclusive) and 1.0")
        if self.vad_offset is not None and not (0.0 < self.vad_offset <= 1.0):
            errors.append("vad_offset must be between 0.0 (exclusive) and 1.0")
        if self.compression_ratio_threshold is not None and self.compression_ratio_threshold <= 0:
            errors.append("compression_ratio_threshold must be > 0")
        if self.log_prob_threshold is not None and self.log_prob_threshold > 0:
            errors.append("log_prob_threshold must be <= 0 (log probabilities are negative)")
        if self.prompt_reset_on_temperature is not None and not (0.0 <= self.prompt_reset_on_temperature <= 1.0):
            errors.append("prompt_reset_on_temperature must be between 0.0 and 1.0")
        if self.max_initial_timestamp is not None and self.max_initial_timestamp < 0:
            errors.append("max_initial_timestamp must be >= 0")
        if self.batch_size is not None and self.batch_size < 1:
            errors.append("batch_size must be >= 1")
        if self.max_new_tokens is not None and self.max_new_tokens < 1:
            errors.append("max_new_tokens must be >= 1")
        if self.hallucination_silence_threshold is not None and self.hallucination_silence_threshold <= 0:
            errors.append("hallucination_silence_threshold must be > 0")
        if self.temperatures is not None:
            try:
                vals = [float(t.strip()) for t in self.temperatures.split(",")]
                for v in vals:
                    if not (0.0 <= v <= 1.0):
                        errors.append("temperatures: each value must be between 0.0 and 1.0")
                        break
            except ValueError:
                errors.append("temperatures must be comma-separated numbers (e.g. '0,0.2,0.4')")
        return errors


# Fields on TranscriptionOptions that we allow per-request mutation of.
_ASR_OPTION_FIELDS = [
    "beam_size", "best_of", "patience", "length_penalty",
    "repetition_penalty", "no_repeat_ngram_size",
    "compression_ratio_threshold", "log_prob_threshold",
    "no_speech_threshold", "condition_on_previous_text",
    "prompt_reset_on_temperature", "suppress_blank",
    "without_timestamps", "max_initial_timestamp",
    "max_new_tokens", "hallucination_silence_threshold",
    "prefix", "prepend_punctuations", "append_punctuations",
]


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------
def clear_gpu_memory():
    """Clear GPU memory cache to prevent VRAM buildup."""
    if DEVICE == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        logger.debug("GPU memory cache cleared")


# ---------------------------------------------------------------------------
# Stage 0 -- model loading
# ---------------------------------------------------------------------------
def load_whisper_model(model_name: str):
    """Load WhisperX model with caching (thread-safe)."""
    if model_name not in _whisper_models:
        with _model_load_lock:
            if model_name not in _whisper_models:
                logger.info(f"Loading WhisperX model: {model_name}")
                model = whisperx.load_model(
                    model_name,
                    device=DEVICE,
                    compute_type=COMPUTE_TYPE,
                    download_root=CACHE_DIR,
                )
                _whisper_models[model_name] = model
                logger.info(f"Model {model_name} loaded successfully")
    return _whisper_models[model_name]


def load_align_model(language_code: str):
    """Load alignment model with per-language caching (thread-safe)."""
    if language_code not in _align_models:
        with _model_load_lock:
            if language_code not in _align_models:
                logger.info(f"Loading alignment model for language: {language_code}")
                model_a, metadata = whisperx.load_align_model(
                    language_code=language_code,
                    device=DEVICE,
                    model_dir=CACHE_DIR,
                )
                _align_models[language_code] = (model_a, metadata)
                logger.info(f"Alignment model for {language_code} loaded")
    return _align_models[language_code]


def load_diarize_pipeline() -> DiarizationPipeline:
    """Load diarization pipeline (singleton, thread-safe)."""
    global _diarize_pipeline
    if _diarize_pipeline is None:
        with _model_load_lock:
            if _diarize_pipeline is None:
                logger.info("Loading diarization pipeline: pyannote/speaker-diarization-3.1")
                _diarize_pipeline = DiarizationPipeline(
                    model_name="pyannote/speaker-diarization-3.1",
                    device=torch.device(DEVICE),
                )
                logger.info("Diarization pipeline loaded")
    return _diarize_pipeline


# ---------------------------------------------------------------------------
# Stage 1 -- Transcription
# ---------------------------------------------------------------------------
def transcribe(
    audio: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    language: Optional[str] = None,
    task: str = "transcribe",
    initial_prompt: Optional[str] = None,
    hotwords: Optional[str] = None,
    params: Optional[TranscribeParams] = None,
) -> dict:
    """Run WhisperX transcription and return raw result dict.

    Per-request overrides are applied via snapshot/mutate/restore on the
    cached model singleton.  All original state is restored in the finally
    block regardless of success or failure.
    """
    whisper_model = load_whisper_model(model_name)

    # Snapshot original state so we can restore after transcription.
    # Only snapshot when we actually have overrides to apply.
    has_overrides = (
        hotwords is not None
        or initial_prompt is not None
        or (params is not None and params.has_overrides())
    )

    if has_overrides:
        original_options = copy.copy(whisper_model.options)
        original_vad_params = dict(whisper_model._vad_params)
        original_suppress_numerals = whisper_model.suppress_numerals
        original_vad_onset = getattr(whisper_model.vad_model, "vad_onset", None)
        original_vad_offset = getattr(whisper_model.vad_model, "vad_offset", None)

    try:
        # Apply per-request overrides
        if hotwords is not None:
            whisper_model.options.hotwords = hotwords
        if initial_prompt is not None:
            whisper_model.options.initial_prompt = initial_prompt

        if params is not None:
            # ASR option overrides (TranscriptionOptions fields)
            for field_name in _ASR_OPTION_FIELDS:
                value = getattr(params, field_name, None)
                if value is not None:
                    setattr(whisper_model.options, field_name, value)

            # temperatures: comma-separated string → list of floats
            if params.temperatures is not None:
                whisper_model.options.temperatures = [
                    float(t.strip()) for t in params.temperatures.split(",")
                ]

            # clip_timestamps: comma-separated string → list of floats
            if params.clip_timestamps is not None:
                whisper_model.options.clip_timestamps = [
                    float(t.strip()) for t in params.clip_timestamps.split(",")
                ]

            # suppress_numerals: instance attribute on FasterWhisperPipeline
            if params.suppress_numerals is not None:
                whisper_model.suppress_numerals = params.suppress_numerals

            # VAD param overrides (used by merge_chunks in transcribe())
            if params.vad_onset is not None:
                whisper_model._vad_params["vad_onset"] = params.vad_onset
                if hasattr(whisper_model.vad_model, "vad_onset"):
                    whisper_model.vad_model.vad_onset = params.vad_onset
            if params.vad_offset is not None:
                whisper_model._vad_params["vad_offset"] = params.vad_offset
                if hasattr(whisper_model.vad_model, "vad_offset"):
                    whisper_model.vad_model.vad_offset = params.vad_offset

        # Build transcribe() call arguments
        transcribe_options: Dict[str, Any] = {
            "batch_size": params.batch_size if params and params.batch_size else BATCH_SIZE,
            "language": language,
            "task": task,
        }
        if params and params.chunk_size is not None:
            transcribe_options["chunk_size"] = params.chunk_size

        logger.info("Starting transcription...")
        result = whisper_model.transcribe(audio, **transcribe_options)

    finally:
        if has_overrides:
            whisper_model.options = original_options
            whisper_model._vad_params = original_vad_params
            whisper_model.suppress_numerals = original_suppress_numerals
            if original_vad_onset is not None and hasattr(whisper_model.vad_model, "vad_onset"):
                whisper_model.vad_model.vad_onset = original_vad_onset
            if original_vad_offset is not None and hasattr(whisper_model.vad_model, "vad_offset"):
                whisper_model.vad_model.vad_offset = original_vad_offset

    detected_language = result.get("language", language or "en")
    logger.info(f"Transcription complete. Detected language: {detected_language}")

    clear_gpu_memory()
    return result


# ---------------------------------------------------------------------------
# Stage 2 -- Alignment
# ---------------------------------------------------------------------------
def align(audio: np.ndarray, result: dict) -> dict:
    """Run Wav2Vec2 alignment to get word-level timestamps."""
    detected_language = result.get("language", "en")
    logger.info("Aligning timestamps...")
    try:
        model_a, metadata = load_align_model(detected_language)
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            DEVICE,
            return_char_alignments=False,
        )
        logger.info("Timestamp alignment complete")
        clear_gpu_memory()
    except Exception as e:
        logger.warning(f"Timestamp alignment failed: {e}, continuing without word-level timestamps")
    return result


# ---------------------------------------------------------------------------
# Stage 3 -- Diarization
# ---------------------------------------------------------------------------
def diarize(
    audio: np.ndarray,
    result: dict,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    return_speaker_embeddings: bool = False,
) -> Tuple[dict, Optional[dict]]:
    """
    Run pyannote speaker diarization and assign speakers to segments.

    Returns (result_with_speakers, speaker_embeddings_or_None).
    """
    if not HF_TOKEN:
        logger.warning("Speaker diarization requested but HF_TOKEN not set")
        return result, None

    logger.info("Starting speaker diarization...")
    speaker_embeddings = None
    try:
        diarize_model = load_diarize_pipeline()

        diarize_params: Dict[str, Any] = {}
        if num_speakers is not None:
            diarize_params["num_speakers"] = num_speakers
            logger.info(f"Diarization with exact speaker count: {num_speakers}")
        else:
            if min_speakers is not None:
                diarize_params["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarize_params["max_speakers"] = max_speakers
            logger.info(f"Diarization with speaker range: {min_speakers}-{max_speakers}")

        if return_speaker_embeddings:
            diarize_params["return_embeddings"] = True
            logger.info("Speaker embeddings will be returned")

        diarize_output = diarize_model(audio, **diarize_params)

        if return_speaker_embeddings and isinstance(diarize_output, tuple):
            diarize_segments, speaker_embeddings = diarize_output
            logger.info(f"Received speaker embeddings for {len(speaker_embeddings)} speakers")
        else:
            diarize_segments = diarize_output

        if hasattr(diarize_segments, "exclusive_speaker_diarization"):
            diarize_segments = diarize_segments.exclusive_speaker_diarization
            logger.info("Using exclusive speaker diarization for better timestamp reconciliation")

        result = whisperx.assign_word_speakers(diarize_segments, result)
        logger.info("Speaker diarization complete")
        clear_gpu_memory()
    except Exception as e:
        logger.warning(f"Speaker diarization failed: {e}, continuing without diarization")

    return result, speaker_embeddings


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------
def sanitize_float_values(obj):
    """Recursively sanitize float values for JSON compliance (NaN/Inf -> None)."""
    if isinstance(obj, dict):
        return {key: sanitize_float_values(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_float_values(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return sanitize_float_values(obj.tolist())
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return obj


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ---------------------------------------------------------------------------
# Convenience: full pipeline in one call
# ---------------------------------------------------------------------------
def run_pipeline(
    audio: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    language: Optional[str] = None,
    task: str = "transcribe",
    initial_prompt: Optional[str] = None,
    hotwords: Optional[str] = None,
    word_timestamps: bool = True,
    should_diarize: bool = True,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    return_speaker_embeddings: bool = False,
    params: Optional[TranscribeParams] = None,
) -> Tuple[dict, Optional[dict]]:
    """
    Run the full 3-stage pipeline: transcribe -> align -> diarize.

    Returns (result, speaker_embeddings_or_None).
    """
    result = transcribe(
        audio,
        model_name=model_name,
        language=language,
        task=task,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
        params=params,
    )

    if word_timestamps:
        result = align(audio, result)

    speaker_embeddings = None
    if should_diarize:
        result, speaker_embeddings = diarize(
            audio,
            result,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            return_speaker_embeddings=return_speaker_embeddings,
        )

    return result, speaker_embeddings
