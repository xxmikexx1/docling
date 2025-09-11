import logging
from enum import Enum

from pydantic import (
    AnyUrl,
)

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.pipeline_options_asr_model import (
    # AsrResponseFormat,
    # ApiAsrOptions,
    InferenceAsrFramework,
    InlineAsrNativeWhisperOptions,
    TransformersModelType,
)

_log = logging.getLogger(__name__)

WHISPER_TINY = InlineAsrNativeWhisperOptions(
    repo_id="tiny",
    inference_framework=InferenceAsrFramework.WHISPER,
    verbose=True,
    timestamps=True,
    word_timestamps=True,
    temperature=0.0,
    max_new_tokens=256,
    max_time_chunk=30.0,
)

WHISPER_SMALL = InlineAsrNativeWhisperOptions(
    repo_id="small",
    inference_framework=InferenceAsrFramework.WHISPER,
    verbose=True,
    timestamps=True,
    word_timestamps=True,
    temperature=0.0,
    max_new_tokens=256,
    max_time_chunk=30.0,
)

WHISPER_MEDIUM = InlineAsrNativeWhisperOptions(
    repo_id="medium",
    inference_framework=InferenceAsrFramework.WHISPER,
    verbose=True,
    timestamps=True,
    word_timestamps=True,
    temperature=0.0,
    max_new_tokens=256,
    max_time_chunk=30.0,
)

WHISPER_BASE = InlineAsrNativeWhisperOptions(
    repo_id="base",
    inference_framework=InferenceAsrFramework.WHISPER,
    verbose=True,
    timestamps=True,
    word_timestamps=True,
    temperature=0.0,
    max_new_tokens=256,
    max_time_chunk=30.0,
)

WHISPER_LARGE = InlineAsrNativeWhisperOptions(
    repo_id="large",
    inference_framework=InferenceAsrFramework.WHISPER,
    verbose=True,
    timestamps=True,
    word_timestamps=True,
    temperature=0.0,
    max_new_tokens=256,
    max_time_chunk=30.0,
)

WHISPER_TURBO = InlineAsrNativeWhisperOptions(
    repo_id="turbo",
    inference_framework=InferenceAsrFramework.WHISPER,
    verbose=True,
    timestamps=True,
    word_timestamps=True,
    temperature=0.0,
    max_new_tokens=256,
    max_time_chunk=30.0,
)


class AsrModelType(str, Enum):
    """An enumeration of supported Automatic Speech Recognition (ASR) models.

    This enum provides a standardized set of identifiers for different ASR models,
    making it easier to specify which model to use for transcription tasks.

    Attributes:
        WHISPER_TINY: The 'tiny' version of the Whisper model.
        WHISPER_SMALL: The 'small' version of the Whisper model.
        WHISPER_MEDIUM: The 'medium' version of the Whisper model.
        WHISPER_BASE: The 'base' version of the Whisper model.
        WHISPER_LARGE: The 'large' version of the Whisper model.
        WHISPER_TURBO: A 'turbo' version of the Whisper model, optimized for speed.
    """

    WHISPER_TINY = "whisper_tiny"
    WHISPER_SMALL = "whisper_small"
    WHISPER_MEDIUM = "whisper_medium"
    WHISPER_BASE = "whisper_base"
    WHISPER_LARGE = "whisper_large"
    WHISPER_TURBO = "whisper_turbo"
