from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from docling_core.types.doc.page import SegmentedPage
from pydantic import AnyUrl, BaseModel
from typing_extensions import deprecated

from docling.datamodel.accelerator_options import AcceleratorDevice


class BaseVlmOptions(BaseModel):
    """A base class for Vision Language Model (VLM) options.

    This class serves as a foundation for VLM-related configuration models,
    providing common attributes such as the prompt, image scaling, and temperature.
    It also includes methods for building the final prompt and decoding the
    model's response.

    Attributes:
        kind: A string identifier for the type of VLM options.
        prompt: The prompt template to be used for the VLM.
        scale: The scaling factor to apply to images before processing.
        max_size: An optional integer specifying the maximum size of the image.
        temperature: The temperature for sampling, controlling the randomness of
            the VLM's output.
    """

    kind: str
    prompt: str
    scale: float = 2.0
    max_size: Optional[int] = None
    temperature: float = 0.0

    def build_prompt(self, page: Optional[SegmentedPage]) -> str:
        """Builds the final prompt to be sent to the VLM.

        This method can be overridden by subclasses to construct a more complex
        prompt, potentially incorporating information from the page content.

        Args:
            page: An optional `SegmentedPage` object containing the parsed
                content of the page.

        Returns:
            The final prompt string.
        """
        return self.prompt

    def decode_response(self, text: str) -> str:
        """Decodes the raw text response from the VLM.

        This method can be overridden by subclasses to perform any necessary
        post-processing on the VLM's output, such as stripping special tokens
        or formatting the text.

        Args:
            text: The raw text output from the VLM.

        Returns:
            The decoded and cleaned text.
        """
        return text


class ResponseFormat(str, Enum):
    """An enumeration of the expected response formats from a VLM.

    This enum provides a standardized set of identifiers for the different
    output formats that a VLM can be prompted to produce.

    Attributes:
        DOCTAGS: The DocTags structured format.
        MARKDOWN: The Markdown format.
        HTML: The HTML format.
        OTSL: The Open-source Table Structure Linter (OTSL) format for tables.
        PLAINTEXT: Plain text format.
    """

    DOCTAGS = "doctags"
    MARKDOWN = "markdown"
    HTML = "html"
    OTSL = "otsl"
    PLAINTEXT = "plaintext"


class InferenceFramework(str, Enum):
    """An enumeration of the supported inference frameworks for VLMs.

    This enum provides a standardized set of identifiers for the different
    frameworks that can be used to run VLM inference.

    Attributes:
        MLX: The MLX framework, optimized for Apple silicon.
        TRANSFORMERS: The Hugging Face Transformers library.
        VLLM: The vLLM framework for fast LLM inference.
    """

    MLX = "mlx"
    TRANSFORMERS = "transformers"
    VLLM = "vllm"


class TransformersModelType(str, Enum):
    """An enumeration of the different AutoModel types from the Hugging Face Transformers library.

    This enum specifies which `AutoModel` class to use when loading a model,
    which is important for ensuring that the correct model architecture is used.

    Attributes:
        AUTOMODEL: The generic `AutoModel` class.
        AUTOMODEL_VISION2SEQ: The `AutoModelForVision2Seq` class for vision-encoder-decoder models.
        AUTOMODEL_CAUSALLM: The `AutoModelForCausalLM` class for causal language models.
        AUTOMODEL_IMAGETEXTTOTEXT: A custom type for image-text-to-text models.
    """

    AUTOMODEL = "automodel"
    AUTOMODEL_VISION2SEQ = "automodel-vision2seq"
    AUTOMODEL_CAUSALLM = "automodel-causallm"
    AUTOMODEL_IMAGETEXTTOTEXT = "automodel-imagetexttotext"


class TransformersPromptStyle(str, Enum):
    """An enumeration for the different prompt styles used with Transformers models.

    This enum specifies how the prompt should be formatted before being passed
    to the model, which is important for models that expect a specific chat
    or instruction format.

    Attributes:
        CHAT: Uses the chat template provided by the model's tokenizer.
        RAW: Uses the raw prompt string without any special formatting.
        NONE: No prompt is used.
    """

    CHAT = "chat"
    RAW = "raw"
    NONE = "none"


class InlineVlmOptions(BaseVlmOptions):
    """Configuration options for running an inline Vision Language Model (VLM).

    This class provides a comprehensive set of settings for using a VLM that
    runs locally. It includes parameters for model loading, quantization,
    inference framework, prompt styling, and generation.

    Attributes:
        kind: A class variable specifying the option type, fixed to "inline_model_options".
        repo_id: The repository ID of the VLM on Hugging Face.
        trust_remote_code: If `True`, allows the execution of remote code from the
            model's repository.
        load_in_8bit: If `True`, loads the model in 8-bit precision.
        llm_int8_threshold: The threshold for 8-bit quantization.
        quantized: If `True`, indicates that the model is quantized.
        inference_framework: The `InferenceFramework` to use for running the model.
        transformers_model_type: The `TransformersModelType` to use when loading
            the model.
        transformers_prompt_style: The `TransformersPromptStyle` to use for
            formatting the prompt.
        response_format: The expected `ResponseFormat` from the VLM.
        torch_dtype: The torch data type to use for the model (e.g., "float16").
        supported_devices: A list of supported `AcceleratorDevice`s.
        stop_strings: A list of strings that will cause the generation to stop.
        extra_generation_config: A dictionary of additional parameters for the
            generation process.
        extra_processor_kwargs: A dictionary of additional keyword arguments for
            the model's processor.
        use_kv_cache: If `True`, uses a key-value cache to speed up generation.
        max_new_tokens: The maximum number of new tokens to generate.
        repo_cache_folder: A property that generates a local folder name for
            caching the model.
    """

    kind: Literal["inline_model_options"] = "inline_model_options"

    repo_id: str
    trust_remote_code: bool = False
    load_in_8bit: bool = True
    llm_int8_threshold: float = 6.0
    quantized: bool = False

    inference_framework: InferenceFramework
    transformers_model_type: TransformersModelType = TransformersModelType.AUTOMODEL
    transformers_prompt_style: TransformersPromptStyle = TransformersPromptStyle.CHAT
    response_format: ResponseFormat

    torch_dtype: Optional[str] = None
    supported_devices: List[AcceleratorDevice] = [
        AcceleratorDevice.CPU,
        AcceleratorDevice.CUDA,
        AcceleratorDevice.MPS,
    ]

    stop_strings: List[str] = []
    extra_generation_config: Dict[str, Any] = {}
    extra_processor_kwargs: Dict[str, Any] = {}

    use_kv_cache: bool = True
    max_new_tokens: int = 4096

    @property
    def repo_cache_folder(self) -> str:
        return self.repo_id.replace("/", "--")


@deprecated("Use InlineVlmOptions instead.")
class HuggingFaceVlmOptions(InlineVlmOptions):
    pass


class ApiVlmOptions(BaseVlmOptions):
    """Configuration options for using an API-based Vision Language Model (VLM).

    This class provides settings for connecting to an external API to use a VLM.
    It includes parameters for the API endpoint, headers, timeout, concurrency,
    and the expected response format.

    Attributes:
        kind: A class variable specifying the option type, fixed to "api_model_options".
        url: The URL of the chat completions API endpoint.
        headers: A dictionary of HTTP headers to include in the API request.
        params: A dictionary of query parameters to include in the API request.
        timeout: The request timeout in seconds.
        concurrency: The number of concurrent requests to make to the API.
        response_format: The expected `ResponseFormat` from the VLM.
    """

    kind: Literal["api_model_options"] = "api_model_options"

    url: AnyUrl = AnyUrl(
        "http://localhost:11434/v1/chat/completions"
    )  # Default to ollama
    headers: Dict[str, str] = {}
    params: Dict[str, Any] = {}
    timeout: float = 60
    concurrency: int = 1
    response_format: ResponseFormat
