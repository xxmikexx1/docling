import logging
import os
import re
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union, cast

from docling_core.types.doc import DoclingDocument, DocumentOrigin

# import whisper  # type: ignore
# import librosa
# import numpy as np
# import soundfile as sf  # type: ignore
from docling_core.types.doc.labels import DocItemLabel
from pydantic import BaseModel, Field, validator

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.noop_backend import NoOpBackend

# from pydub import AudioSegment  # type: ignore
# from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from docling.datamodel.accelerator_options import (
    AcceleratorOptions,
)
from docling.datamodel.base_models import (
    ConversionStatus,
    FormatToMimeType,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import (
    AsrPipelineOptions,
)
from docling.datamodel.pipeline_options_asr_model import (
    InlineAsrNativeWhisperOptions,
    # AsrResponseFormat,
    InlineAsrOptions,
)
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
)
from docling.datamodel.settings import settings
from docling.pipeline.base_pipeline import BasePipeline
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import ProfilingScope, TimeRecorder

_log = logging.getLogger(__name__)


class _ConversationWord(BaseModel):
    """Represents a single word in a conversation transcript.

    Attributes:
        text: The text of the word.
        start_time: The start time of the word in seconds from the beginning of the audio.
        end_time: The end time of the word in seconds from the beginning of the audio.
    """

    text: str
    start_time: Optional[float] = Field(
        None, description="Start time in seconds from video start"
    )
    end_time: Optional[float] = Field(
        None, ge=0, description="End time in seconds from video start"
    )


class _ConversationItem(BaseModel):
    """Represents a segment of a conversation, typically a sentence or a phrase.

    Attributes:
        text: The transcribed text of the conversation segment.
        start_time: The start time of the segment in seconds.
        end_time: The end time of the segment in seconds.
        speaker_id: A numeric identifier for the speaker.
        speaker: The name of the speaker.
        words: A list of individual words with timestamps within this segment.
    """

    text: str
    start_time: Optional[float] = Field(
        None, description="Start time in seconds from video start"
    )
    end_time: Optional[float] = Field(
        None, ge=0, description="End time in seconds from video start"
    )
    speaker_id: Optional[int] = Field(None, description="Numeric speaker identifier")
    speaker: Optional[str] = Field(
        None, description="Speaker name, defaults to speaker-{speaker_id}"
    )
    words: Optional[list[_ConversationWord]] = Field(
        None, description="Individual words with time-stamps"
    )

    def __lt__(self, other):
        """Compares two _ConversationItem objects based on their start time."""
        if not isinstance(other, _ConversationItem):
            return NotImplemented
        return self.start_time < other.start_time

    def __eq__(self, other):
        """Checks if two _ConversationItem objects are equal based on their start time."""
        if not isinstance(other, _ConversationItem):
            return NotImplemented
        return self.start_time == other.start_time

    def to_string(self) -> str:
        """Formats the conversation entry as a string.

        Returns:
            A string representation of the conversation item, including time and speaker information if available.
        """
        result = ""
        if (self.start_time is not None) and (self.end_time is not None):
            result += f"[time: {self.start_time}-{self.end_time}] "

        if self.speaker is not None:
            result += f"[speaker:{self.speaker}] "

        result += self.text
        return result


class _NativeWhisperModel:
    """A wrapper for the OpenAI Whisper model for audio transcription.

    This class handles the initialization of the Whisper model and provides
    methods to run the transcription process on an audio file.

    Attributes:
        enabled: A boolean indicating whether the model is enabled.
        asr_options: Configuration options for the ASR model.
        max_tokens: The maximum number of tokens to generate.
        temperature: The sampling temperature for the model.
        device: The device (CPU or GPU) to run the model on.
        model_name: The name of the Whisper model to use.
        model: The loaded Whisper model instance.
        verbose: A boolean indicating whether to print verbose output.
        timestamps: A boolean indicating whether to generate timestamps.
        word_timestamps: A boolean indicating whether to generate word-level timestamps.
    """

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        asr_options: InlineAsrNativeWhisperOptions,
    ):
        """Initializes the _NativeWhisperModel.

        Args:
            enabled: Whether the model should be enabled.
            artifacts_path: The path to the model artifacts.
            accelerator_options: Options for hardware acceleration.
            asr_options: Options for the ASR model.
        """
        self.enabled = enabled

        _log.info(f"artifacts-path: {artifacts_path}")
        _log.info(f"accelerator_options: {accelerator_options}")

        if self.enabled:
            try:
                import whisper  # type: ignore
            except ImportError:
                raise ImportError(
                    "whisper is not installed. Please install it via `pip install openai-whisper` or do `uv sync --extra asr`."
                )
            self.asr_options = asr_options
            self.max_tokens = asr_options.max_new_tokens
            self.temperature = asr_options.temperature

            self.device = decide_device(
                accelerator_options.device,
                supported_devices=asr_options.supported_devices,
            )
            _log.info(f"Available device for Whisper: {self.device}")

            self.model_name = asr_options.repo_id
            _log.info(f"loading _NativeWhisperModel({self.model_name})")
            if artifacts_path is not None:
                _log.info(f"loading {self.model_name} from {artifacts_path}")
                self.model = whisper.load_model(
                    name=self.model_name,
                    device=self.device,
                    download_root=str(artifacts_path),
                )
            else:
                self.model = whisper.load_model(
                    name=self.model_name, device=self.device
                )

            self.verbose = asr_options.verbose
            self.timestamps = asr_options.timestamps
            self.word_timestamps = asr_options.word_timestamps

    def run(self, conv_res: ConversionResult) -> ConversionResult:
        """Runs the audio transcription process.

        Args:
            conv_res: The ConversionResult object containing the input audio file.

        Returns:
            The updated ConversionResult object with the transcription results.
        """
        audio_path: Path = Path(conv_res.input.file).resolve()

        try:
            conversation = self.transcribe(audio_path)

            # Ensure we have a proper DoclingDocument
            origin = DocumentOrigin(
                filename=conv_res.input.file.name or "audio.wav",
                mimetype="audio/x-wav",
                binary_hash=conv_res.input.document_hash,
            )
            conv_res.document = DoclingDocument(
                name=conv_res.input.file.stem or "audio.wav", origin=origin
            )

            for citem in conversation:
                conv_res.document.add_text(
                    label=DocItemLabel.TEXT, text=citem.to_string()
                )

            conv_res.status = ConversionStatus.SUCCESS
            return conv_res

        except Exception as exc:
            _log.error(f"Audio tranciption has an error: {exc}")

        conv_res.status = ConversionStatus.FAILURE
        return conv_res

    def transcribe(self, fpath: Path) -> list[_ConversationItem]:
        """Transcribes an audio file using the Whisper model.

        Args:
            fpath: The path to the audio file.

        Returns:
            A list of _ConversationItem objects representing the transcribed conversation.
        """
        result = self.model.transcribe(
            str(fpath), verbose=self.verbose, word_timestamps=self.word_timestamps
        )

        convo: list[_ConversationItem] = []
        for _ in result["segments"]:
            item = _ConversationItem(
                start_time=_["start"], end_time=_["end"], text=_["text"], words=[]
            )
            if "words" in _ and self.word_timestamps:
                item.words = []
                for __ in _["words"]:
                    item.words.append(
                        _ConversationWord(
                            start_time=__["start"],
                            end_time=__["end"],
                            text=__["word"],
                        )
                    )
            convo.append(item)

        return convo


class AsrPipeline(BasePipeline):
    """A pipeline for performing Automatic Speech Recognition (ASR) on audio files.

    This pipeline uses a configured ASR model (e.g., Whisper) to transcribe
    the audio and create a DoclingDocument with the transcribed text.

    Attributes:
        pipeline_options: The configuration options for this pipeline.
        _model: The ASR model instance.
    """

    def __init__(self, pipeline_options: AsrPipelineOptions):
        """Initializes the AsrPipeline.

        Args:
            pipeline_options: The options for configuring the pipeline.
        """
        super().__init__(pipeline_options)
        self.keep_backend = True

        self.pipeline_options: AsrPipelineOptions = pipeline_options

        if isinstance(self.pipeline_options.asr_options, InlineAsrNativeWhisperOptions):
            asr_options: InlineAsrNativeWhisperOptions = (
                self.pipeline_options.asr_options
            )
            self._model = _NativeWhisperModel(
                enabled=True,  # must be always enabled for this pipeline to make sense.
                artifacts_path=self.artifacts_path,
                accelerator_options=pipeline_options.accelerator_options,
                asr_options=asr_options,
            )
        else:
            _log.error(f"No model support for {self.pipeline_options.asr_options}")

    def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
        """Determines the final status of the conversion.

        Returns:
            The final status of the conversion.
        """
        status = ConversionStatus.SUCCESS
        return status

    @classmethod
    def get_default_options(cls) -> AsrPipelineOptions:
        """Returns the default options for this pipeline.

        Returns:
            The default AsrPipelineOptions.
        """
        return AsrPipelineOptions()

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Builds the document by running the ASR model.

        Args:
            conv_res: The ConversionResult object.

        Returns:
            The updated ConversionResult with the transcribed document.
        """
        _log.info(f"start _build_document in AsrPipeline: {conv_res.input.file}")
        with TimeRecorder(conv_res, "doc_build", scope=ProfilingScope.DOCUMENT):
            self._model.run(conv_res=conv_res)

        return conv_res

    @classmethod
    def is_backend_supported(cls, backend: AbstractDocumentBackend):
        """Checks if a given backend is supported by this pipeline.

        Args:
            backend: The document backend to check.

        Returns:
            True if the backend is supported, False otherwise.
        """
        return isinstance(backend, NoOpBackend)
