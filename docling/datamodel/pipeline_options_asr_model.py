from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import AnyUrl, BaseModel
from typing_extensions import deprecated

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.pipeline_options_vlm_model import (
    # InferenceFramework,
    TransformersModelType,
)


class BaseAsrOptions(BaseModel):
    """A base class for Automatic Speech Recognition (ASR) options.

    This class serves as a foundation for ASR-related configuration models,
    providing a common `kind` attribute to identify the type of options.
    """

    kind: str


class InferenceAsrFramework(str, Enum):
    """An enumeration of the supported inference frameworks for ASR models.

    This enum provides a standardized set of identifiers for the different
    frameworks that can be used to run ASR model inference.

    Attributes:
        WHISPER: The native Whisper inference framework.
    """

    # MLX = "mlx" # disabled for now
    # TRANSFORMERS = "transformers" # disabled for now
    WHISPER = "whisper"


class InlineAsrOptions(BaseAsrOptions):
    """Configuration options for running an inline ASR model.

    This class provides settings for using an ASR model that runs locally,
    including parameters for the model repository, generation settings, and
    hardware acceleration.

    Attributes:
        kind: A class variable specifying the option type, fixed to "inline_model_options".
        repo_id: The repository ID of the ASR model on Hugging Face.
        verbose: If `True`, enables verbose output during transcription.
        timestamps: If `True`, includes timestamps for transcribed segments.
        temperature: The temperature for sampling, controlling the randomness of
            the output.
        max_new_tokens: The maximum number of tokens to generate.
        max_time_chunk: The maximum duration of an audio chunk to process at once.
        torch_dtype: The torch data type to use for the model (e.g., "float16").
        supported_devices: A list of `AcceleratorDevice` enums indicating which
            hardware accelerators are compatible with this model.
        repo_cache_folder: A property that generates a local folder name for
            caching the model.
    """

    kind: Literal["inline_model_options"] = "inline_model_options"

    repo_id: str

    verbose: bool = False
    timestamps: bool = True

    temperature: float = 0.0
    max_new_tokens: int = 256
    max_time_chunk: float = 30.0

    torch_dtype: Optional[str] = None
    supported_devices: List[AcceleratorDevice] = [
        AcceleratorDevice.CPU,
        AcceleratorDevice.CUDA,
        AcceleratorDevice.MPS,
    ]

    @property
    def repo_cache_folder(self) -> str:
        return self.repo_id.replace("/", "--")


class InlineAsrNativeWhisperOptions(InlineAsrOptions):
    """Configuration options specific to the native Whisper ASR model.

    This class extends `InlineAsrOptions` with settings that are particular to
    the Whisper model when used with its native inference framework.

    Attributes:
        inference_framework: The inference framework, fixed to `WHISPER`.
        language: The language of the audio to be transcribed.
        supported_devices: A list of supported hardware accelerators, typically
            CPU and CUDA for the native Whisper framework.
        word_timestamps: If `True`, generates timestamps for individual words
            in the transcription.
    """

    inference_framework: InferenceAsrFramework = InferenceAsrFramework.WHISPER

    language: str = "en"
    supported_devices: List[AcceleratorDevice] = [
        AcceleratorDevice.CPU,
        AcceleratorDevice.CUDA,
    ]
    word_timestamps: bool = True
