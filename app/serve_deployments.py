"""
Ray Serve deployment classes for the 3-stage WhisperX pipeline.

Each stage is an independent deployment with @serve.batch for cross-request
batching and fractional GPU allocation.
"""

import os
import logging
from typing import Optional, List, Tuple

import numpy as np
from ray import serve

from app.pipeline import (
    transcribe as _transcribe,
    align as _align,
    diarize as _diarize,
    load_whisper_model,
    load_align_model,
    load_diarize_pipeline,
    DEFAULT_MODEL,
    HF_TOKEN,
)

logger = logging.getLogger(__name__)

# Batch configuration from env
WHISPER_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "4"))
ALIGN_BATCH_SIZE = int(os.getenv("ALIGN_BATCH_SIZE", "8"))
DIARIZE_BATCH_SIZE = int(os.getenv("DIARIZE_BATCH_SIZE", "2"))
BATCH_WAIT_TIMEOUT = float(os.getenv("BATCH_WAIT_TIMEOUT", "0.1"))


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": float(os.getenv("WHISPER_GPU_FRACTION", "0.5"))},
)
class WhisperDeployment:
    """Stage 1: Transcription via WhisperX."""

    def __init__(self):
        preload = os.getenv("PRELOAD_MODEL", None)
        if preload:
            logger.info(f"WhisperDeployment: preloading model {preload}")
            load_whisper_model(preload)

    @serve.batch(max_batch_size=WHISPER_BATCH_SIZE, batch_wait_timeout_s=BATCH_WAIT_TIMEOUT)
    async def transcribe_batch(
        self,
        audios: List[np.ndarray],
        model_names: List[str],
        languages: List[Optional[str]],
        tasks: List[str],
        initial_prompts: List[Optional[str]],
    ) -> List[dict]:
        results = []
        for audio, model_name, language, task, prompt in zip(
            audios, model_names, languages, tasks, initial_prompts
        ):
            result = _transcribe(
                audio,
                model_name=model_name,
                language=language,
                task=task,
                initial_prompt=prompt,
            )
            results.append(result)
        return results

    async def transcribe(
        self,
        audio: np.ndarray,
        model_name: str = DEFAULT_MODEL,
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
    ) -> dict:
        return await self.transcribe_batch(
            audio, model_name, language, task, initial_prompt
        )


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": float(os.getenv("ALIGN_GPU_FRACTION", "0.3"))},
)
class AlignDeployment:
    """Stage 2: Wav2Vec2 word-level alignment."""

    @serve.batch(max_batch_size=ALIGN_BATCH_SIZE, batch_wait_timeout_s=BATCH_WAIT_TIMEOUT)
    async def align_batch(
        self,
        audios: List[np.ndarray],
        results: List[dict],
    ) -> List[dict]:
        aligned = []
        for audio, result in zip(audios, results):
            aligned.append(_align(audio, result))
        return aligned

    async def align(self, audio: np.ndarray, result: dict) -> dict:
        return await self.align_batch(audio, result)


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": float(os.getenv("DIARIZE_GPU_FRACTION", "0.2"))},
)
class DiarizeDeployment:
    """Stage 3: Pyannote speaker diarization."""

    def __init__(self):
        if HF_TOKEN:
            logger.info("DiarizeDeployment: preloading diarization pipeline")
            load_diarize_pipeline()

    @serve.batch(
        max_batch_size=DIARIZE_BATCH_SIZE,
        batch_wait_timeout_s=float(os.getenv("DIARIZE_BATCH_WAIT_TIMEOUT", "0.2")),
    )
    async def diarize_batch(
        self,
        audios: List[np.ndarray],
        results: List[dict],
        num_speakers_list: List[Optional[int]],
        min_speakers_list: List[Optional[int]],
        max_speakers_list: List[Optional[int]],
        return_embeddings_list: List[bool],
    ) -> List[Tuple[dict, Optional[dict]]]:
        outputs = []
        for audio, result, num_spk, min_spk, max_spk, ret_emb in zip(
            audios, results, num_speakers_list, min_speakers_list,
            max_speakers_list, return_embeddings_list,
        ):
            out = _diarize(
                audio, result,
                num_speakers=num_spk,
                min_speakers=min_spk,
                max_speakers=max_spk,
                return_speaker_embeddings=ret_emb,
            )
            outputs.append(out)
        return outputs

    async def diarize(
        self,
        audio: np.ndarray,
        result: dict,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_speaker_embeddings: bool = False,
    ) -> Tuple[dict, Optional[dict]]:
        return await self.diarize_batch(
            audio, result, num_speakers, min_speakers,
            max_speakers, return_speaker_embeddings,
        )
