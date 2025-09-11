import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
)
from typing_extensions import deprecated

from docling.datamodel import asr_model_specs

# Import the following for backwards compatibility
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.layout_model_specs import (
    DOCLING_LAYOUT_EGRET_LARGE,
    DOCLING_LAYOUT_EGRET_MEDIUM,
    DOCLING_LAYOUT_EGRET_XLARGE,
    DOCLING_LAYOUT_HERON,
    DOCLING_LAYOUT_HERON_101,
    DOCLING_LAYOUT_V2,
    LayoutModelConfig,
)
from docling.datamodel.pipeline_options_asr_model import (
    InlineAsrOptions,
)
from docling.datamodel.pipeline_options_vlm_model import (
    ApiVlmOptions,
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
)
from docling.datamodel.vlm_model_specs import (
    GRANITE_VISION_OLLAMA as granite_vision_vlm_ollama_conversion_options,
    GRANITE_VISION_TRANSFORMERS as granite_vision_vlm_conversion_options,
    NU_EXTRACT_2B_TRANSFORMERS,
    SMOLDOCLING_MLX as smoldocling_vlm_mlx_conversion_options,
    SMOLDOCLING_TRANSFORMERS as smoldocling_vlm_conversion_options,
    VlmModelType,
)

_log = logging.getLogger(__name__)


class BaseOptions(BaseModel):
    """A base class for creating structured option groups.

    This class serves as a foundation for all option-related models in Docling.
    It includes a `kind` attribute, which is used to identify the specific type
    of options being defined.
    """

    kind: ClassVar[str]


class TableFormerMode(str, Enum):
    """An enumeration for the operational modes of the TableFormer model.

    This enum allows for selecting between different performance profiles for
    the TableFormer model, trading off between speed and accuracy.

    Attributes:
        FAST: A mode optimized for faster processing, which may be less accurate.
        ACCURATE: A mode optimized for higher accuracy, which may be slower.
    """

    FAST = "fast"
    ACCURATE = "accurate"


class TableStructureOptions(BaseModel):
    """Configuration options for table structure recognition.

    This class provides settings to control how table structures are identified
    and processed by the TableFormer model.

    Attributes:
        do_cell_matching: A boolean flag that determines whether to match the
            model's predictions back to the text cells extracted from the PDF.
            If `True`, it aligns predictions with existing PDF cells, which can
            fail if cells are merged across columns. If `False`, the model
            defines the text cells independently.
        mode: The `TableFormerMode` to use, allowing a choice between "fast"
            and "accurate" processing.
    """

    do_cell_matching: bool = (
        True
        # True:  Matches predictions back to PDF cells. Can break table output if PDF cells
        #        are merged across table columns.
        # False: Let table structure model define the text cells, ignore PDF cells.
    )
    mode: TableFormerMode = TableFormerMode.ACCURATE


class OcrOptions(BaseOptions):
    """A base class for Optical Character Recognition (OCR) options.

    This class defines common settings applicable to all OCR engines, such as
    language support and whether to force OCR on the entire page.

    Attributes:
        lang: A list of language codes for the OCR engine to use.
        force_full_page_ocr: If `True`, forces OCR to be performed on the
            entire page, even if text is programmatically available.
        bitmap_area_threshold: The percentage of a page's area that must be
            covered by bitmap images to trigger OCR processing.
    """

    lang: List[str]
    force_full_page_ocr: bool = False  # If enabled a full page OCR is always applied
    bitmap_area_threshold: float = (
        0.05  # percentage of the area for a bitmap to processed with OCR
    )


class RapidOcrOptions(OcrOptions):
    """Configuration options for the RapidOCR engine.

    This class provides detailed settings for controlling the behavior of the
    RapidOCR engine, including backend selection, model paths, and various
    performance-tuning parameters.

    Attributes:
        kind: A class variable specifying the OCR engine type, fixed to "rapidocr".
        lang: A list of supported languages. Note: This is not yet fully
            supported by the underlying RapidOCR library.
        backend: The computation backend to use (e.g., "onnxruntime", "openvino").
        text_score: The confidence threshold for recognized text.
        use_det: A boolean to enable or disable text detection.
        use_cls: A boolean to enable or disable text angle classification.
        use_rec: A boolean to enable or disable text recognition.
        print_verbose: If `True`, prints verbose output from the OCR engine.
        det_model_path: The path to a custom detection model.
        cls_model_path: The path to a custom classification model.
        rec_model_path: The path to a custom recognition model.
        rec_keys_path: The path to a custom recognition keys file.
        rec_font_path: The path to a font file for visualization.
    """

    kind: ClassVar[Literal["rapidocr"]] = "rapidocr"

    # English and chinese are the most commly used models and have been tested with RapidOCR.
    lang: List[str] = [
        "english",
        "chinese",
    ]
    # However, language as a parameter is not supported by rapidocr yet
    # and hence changing this options doesn't affect anything.

    # For more details on supported languages by RapidOCR visit
    # https://rapidai.github.io/RapidOCRDocs/blog/2022/09/28/%E6%94%AF%E6%8C%81%E8%AF%86%E5%88%AB%E8%AF%AD%E8%A8%80/

    # For more details on the following options visit
    # https://rapidai.github.io/RapidOCRDocs/install_usage/api/RapidOCR/

    # https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/usage/#__tabbed_3_4
    backend: Literal["onnxruntime", "openvino", "paddle", "torch"] = "onnxruntime"
    text_score: float = 0.5  # same default as rapidocr

    use_det: Optional[bool] = None  # same default as rapidocr
    use_cls: Optional[bool] = None  # same default as rapidocr
    use_rec: Optional[bool] = None  # same default as rapidocr

    print_verbose: bool = False  # same default as rapidocr

    det_model_path: Optional[str] = None  # same default as rapidocr
    cls_model_path: Optional[str] = None  # same default as rapidocr
    rec_model_path: Optional[str] = None  # same default as rapidocr
    rec_keys_path: Optional[str] = None  # same default as rapidocr
    rec_font_path: Optional[str] = None  # same default as rapidocr

    model_config = ConfigDict(
        extra="forbid",
    )


class EasyOcrOptions(OcrOptions):
    """Configuration options for the EasyOCR engine.

    This class provides settings for controlling the EasyOCR engine, including
    language selection, GPU usage, and model storage configuration.

    Attributes:
        kind: A class variable specifying the OCR engine type, fixed to "easyocr".
        lang: A list of language codes to be used by the engine (e.g., "en", "fr").
        use_gpu: An optional boolean to enable or disable GPU acceleration.
        confidence_threshold: The minimum confidence score for recognized text.
        model_storage_directory: The directory where EasyOCR models are stored.
        recog_network: The recognition network to use (e.g., "standard").
        download_enabled: If `True`, allows the engine to download missing models.
        suppress_mps_warnings: If `True`, suppresses warnings related to Metal
            Performance Shaders (MPS) on Apple silicon.
    """

    kind: ClassVar[Literal["easyocr"]] = "easyocr"
    lang: List[str] = ["fr", "de", "es", "en"]

    use_gpu: Optional[bool] = None

    confidence_threshold: float = 0.5

    model_storage_directory: Optional[str] = None
    recog_network: Optional[str] = "standard"
    download_enabled: bool = True

    suppress_mps_warnings: bool = True

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )


class TesseractCliOcrOptions(OcrOptions):
    """Configuration options for the Tesseract command-line interface (CLI) engine.

    This class provides settings for using the Tesseract OCR engine via its
    command-line interface. It allows for specifying languages and the path to
    the Tesseract executable.

    Attributes:
        kind: A class variable specifying the OCR engine type, fixed to "tesseract".
        lang: A list of language codes for Tesseract (e.g., "eng", "fra").
        tesseract_cmd: The command to execute Tesseract (e.g., "tesseract").
        path: An optional path to the Tesseract data files (`--tessdata-dir`).
    """

    kind: ClassVar[Literal["tesseract"]] = "tesseract"
    lang: List[str] = ["fra", "deu", "spa", "eng"]
    tesseract_cmd: str = "tesseract"
    path: Optional[str] = None

    model_config = ConfigDict(
        extra="forbid",
    )


class TesseractOcrOptions(OcrOptions):
    """Configuration options for the Tesseract engine via the `tesserocr` library.

    This class provides settings for using the Tesseract OCR engine through the
    `tesserocr` Python wrapper. It allows for specifying languages and the path
    to the Tesseract data files.

    Attributes:
        kind: A class variable specifying the OCR engine type, fixed to "tesserocr".
        lang: A list of language codes for Tesseract (e.g., "eng", "fra").
        path: An optional path to the Tesseract data files (`--tessdata-dir`).
    """

    kind: ClassVar[Literal["tesserocr"]] = "tesserocr"
    lang: List[str] = ["fra", "deu", "spa", "eng"]
    path: Optional[str] = None

    model_config = ConfigDict(
        extra="forbid",
    )


class OcrMacOptions(OcrOptions):
    """Configuration options for the macOS native OCR engine.

    This class provides settings for using the OCR capabilities built into
    macOS. It allows for specifying languages and the recognition level.

    Attributes:
        kind: A class variable specifying the OCR engine type, fixed to "ocrmac".
        lang: A list of language codes supported by macOS (e.g., "en-US", "fr-FR").
        recognition: The recognition level, either "accurate" or "fast".
        framework: The underlying framework to use, typically "vision".
    """

    kind: ClassVar[Literal["ocrmac"]] = "ocrmac"
    lang: List[str] = ["fr-FR", "de-DE", "es-ES", "en-US"]
    recognition: str = "accurate"
    framework: str = "vision"

    model_config = ConfigDict(
        extra="forbid",
    )


class PictureDescriptionBaseOptions(BaseOptions):
    """A base class for picture description options.

    This class defines common settings for models that generate descriptions
    for pictures, such as batch size, image scaling, and the area threshold
    for processing.

    Attributes:
        batch_size: The number of pictures to process in a single batch.
        scale: The scaling factor to apply to images before processing.
        picture_area_threshold: The minimum percentage of the page area a
            picture must occupy to be processed.
    """

    batch_size: int = 8
    scale: float = 2

    picture_area_threshold: float = (
        0.05  # percentage of the area for a picture to processed with the models
    )


class PictureDescriptionApiOptions(PictureDescriptionBaseOptions):
    """Configuration options for using an API-based picture description service.

    This class provides settings for connecting to an external API to generate
    picture descriptions. It includes parameters for the API endpoint, headers,
    timeout, and concurrency.

    Attributes:
        kind: A class variable specifying the option type, fixed to "api".
        url: The URL of the chat completions API endpoint.
        headers: A dictionary of HTTP headers to include in the API request.
        params: A dictionary of query parameters to include in the API request.
        timeout: The request timeout in seconds.
        concurrency: The number of concurrent requests to make to the API.
        prompt: The prompt to send to the API to request a description.
        provenance: A string to identify the source of the description.
    """

    kind: ClassVar[Literal["api"]] = "api"

    url: AnyUrl = AnyUrl("http://localhost:8000/v1/chat/completions")
    headers: Dict[str, str] = {}
    params: Dict[str, Any] = {}
    timeout: float = 20
    concurrency: int = 1

    prompt: str = "Describe this image in a few sentences."
    provenance: str = ""


class PictureDescriptionVlmOptions(PictureDescriptionBaseOptions):
    """Configuration options for using a local Vision Language Model (VLM) for picture description.

    This class provides settings for using a VLM hosted locally or from a
    Hugging Face repository to generate picture descriptions. It includes
    parameters for the model repository, prompt, and generation configuration.

    Attributes:
        kind: A class variable specifying the option type, fixed to "vlm".
        repo_id: The repository ID of the VLM on Hugging Face.
        prompt: The prompt to use for generating the description.
        generation_config: A dictionary of parameters to control the text
            generation process (e.g., `max_new_tokens`, `do_sample`).
        repo_cache_folder: A property that generates a local folder name for
            caching the model.
    """

    kind: ClassVar[Literal["vlm"]] = "vlm"

    repo_id: str
    prompt: str = "Describe this image in a few sentences."
    # Config from here https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig
    generation_config: Dict[str, Any] = dict(max_new_tokens=200, do_sample=False)

    @property
    def repo_cache_folder(self) -> str:
        return self.repo_id.replace("/", "--")


# SmolVLM
smolvlm_picture_description = PictureDescriptionVlmOptions(
    repo_id="HuggingFaceTB/SmolVLM-256M-Instruct"
)

# GraniteVision
granite_picture_description = PictureDescriptionVlmOptions(
    repo_id="ibm-granite/granite-vision-3.3-2b",
    prompt="What is shown in this image?",
)


# Define an enum for the backend options
class PdfBackend(str, Enum):
    """An enumeration of the supported PDF processing backends.

    This enum provides a standardized set of identifiers for the different
    backends that can be used to parse and process PDF documents.

    Attributes:
        PYPDFIUM2: The `pypdfium2` backend, based on the PDFium library.
        DLPARSE_V1: The version 1 of the Deep Search Parse backend.
        DLPARSE_V2: The version 2 of the Deep Search Parse backend.
        DLPARSE_V4: The version 4 of the Deep Search Parse backend.
    """

    PYPDFIUM2 = "pypdfium2"
    DLPARSE_V1 = "dlparse_v1"
    DLPARSE_V2 = "dlparse_v2"
    DLPARSE_V4 = "dlparse_v4"


# Define an enum for the ocr engines
@deprecated(
    "Use get_ocr_factory().registered_kind to get a list of registered OCR engines."
)
class OcrEngine(str, Enum):
    """An enumeration of the supported Optical Character Recognition (OCR) engines.

    This enum provides a standardized set of identifiers for the different OCR
    engines that can be used within Docling.

    Attributes:
        EASYOCR: The EasyOCR engine.
        TESSERACT_CLI: The Tesseract OCR engine, used via its command-line interface.
        TESSERACT: The Tesseract OCR engine, used via the `tesserocr` library.
        OCRMAC: The native OCR engine available on macOS.
        RAPIDOCR: The RapidOCR engine.
    """

    EASYOCR = "easyocr"
    TESSERACT_CLI = "tesseract_cli"
    TESSERACT = "tesseract"
    OCRMAC = "ocrmac"
    RAPIDOCR = "rapidocr"


class PipelineOptions(BaseOptions):
    """A base class for pipeline configuration options.

    This class defines a set of common options that are applicable to all
    processing pipelines in Docling. It includes settings for timeouts,
    hardware acceleration, and security controls.

    Attributes:
        document_timeout: The maximum time in seconds to spend processing a
            single document.
        accelerator_options: An `AcceleratorOptions` object that specifies the
            hardware acceleration settings to use.
        enable_remote_services: If `True`, allows the pipeline to access remote
            services (e.g., for API-based models).
        allow_external_plugins: If `True`, allows the pipeline to load and use
            external plugins.
        artifacts_path: An optional path to a directory where intermediate
            artifacts and logs should be saved.
    """

    document_timeout: Optional[float] = None
    accelerator_options: AcceleratorOptions = AcceleratorOptions()
    enable_remote_services: bool = False
    allow_external_plugins: bool = False
    artifacts_path: Optional[Union[Path, str]] = None


class ConvertPipelineOptions(PipelineOptions):
    """A base class for conversion-focused pipeline options.

    This class extends `PipelineOptions` with settings that are specific to
    document conversion tasks, such as enabling picture classification and
    description.

    Attributes:
        do_picture_classification: If `True`, enables the classification of
            pictures found in the document.
        do_picture_description: If `True`, enables the generation of textual
            descriptions for pictures.
        picture_description_options: A `PictureDescriptionBaseOptions` object
            that specifies the configuration for the picture description model.
    """

    do_picture_classification: bool = False  # True: classify pictures in documents

    do_picture_description: bool = False  # True: run describe pictures in documents
    picture_description_options: PictureDescriptionBaseOptions = (
        smolvlm_picture_description
    )


class PaginatedPipelineOptions(ConvertPipelineOptions):
    """Options for pipelines that process paginated documents.

    This class extends `ConvertPipelineOptions` with settings that are relevant
    for documents that have pages, such as PDFs. It includes options for
    controlling the generation of images for pages and pictures.

    Attributes:
        images_scale: The scaling factor to apply when generating images of
            pages or pictures.
        generate_page_images: If `True`, generates an image for each page.
        generate_picture_images: If `True`, generates an image for each
            detected picture.
    """

    images_scale: float = 1.0
    generate_page_images: bool = False
    generate_picture_images: bool = False


class VlmPipelineOptions(PaginatedPipelineOptions):
    """Configuration options for the Vision Language Model (VLM) pipeline.

    This class extends `PaginatedPipelineOptions` with settings that are specific
    to pipelines that use VLMs for processing. It includes options for forcing
    the use of backend text and configuring the VLM itself.

    Attributes:
        generate_page_images: A boolean that is `True` by default for VLM
            pipelines, as images are required for VLM processing.
        force_backend_text: If `True`, the text extracted by the backend is
            used instead of text generated by the VLM.
        vlm_options: A `Union[InlineVlmOptions, ApiVlmOptions]` object that
            specifies the configuration for the VLM to be used.
    """

    generate_page_images: bool = True
    force_backend_text: bool = (
        False  # (To be used with vlms, or other generative models)
    )
    # If True, text from backend will be used instead of generated text
    vlm_options: Union[InlineVlmOptions, ApiVlmOptions] = (
        smoldocling_vlm_conversion_options
    )


class LayoutOptions(BaseModel):
    """Configuration options for layout analysis processing.

    This class provides settings for controlling how layout analysis is performed,
    including the handling of orphaned cells, empty clusters, and the choice of
    layout model.

    Attributes:
        create_orphan_clusters: If `True`, creates clusters for text cells that
            were not assigned to any layout region by the model.
        keep_empty_clusters: If `True`, retains clusters that do not contain any
            text cells.
        model_spec: A `LayoutModelConfig` object that specifies which layout
            model to use.
        skip_cell_assignment: If `True`, skips the assignment of text cells to
            clusters, which is useful for VLM-only processing where layout is
            inferred differently.
    """

    create_orphan_clusters: bool = True  # Whether to create clusters for orphaned cells
    keep_empty_clusters: bool = (
        False  # Whether to keep clusters that contain no text cells
    )
    model_spec: LayoutModelConfig = DOCLING_LAYOUT_HERON
    skip_cell_assignment: bool = (
        False  # Skip cell-to-cluster assignment for VLM-only processing
    )


class AsrPipelineOptions(PipelineOptions):
    """Configuration options for the Automatic Speech Recognition (ASR) pipeline.

    This class extends `PipelineOptions` with settings specific to the ASR
    pipeline, allowing for the configuration of the ASR model to be used.

    Attributes:
        asr_options: An `InlineAsrOptions` object that specifies the
            configuration for the ASR model.
    """

    asr_options: Union[InlineAsrOptions] = asr_model_specs.WHISPER_TINY


class VlmExtractionPipelineOptions(PipelineOptions):
    """Configuration options for the VLM-based extraction pipeline.

    This class extends `PipelineOptions` with settings specific to using a
    Vision Language Model for structured data extraction.

    Attributes:
        vlm_options: An `InlineVlmOptions` object that specifies the
            configuration for the VLM to be used for extraction.
    """

    vlm_options: Union[InlineVlmOptions] = NU_EXTRACT_2B_TRANSFORMERS


class PdfPipelineOptions(PaginatedPipelineOptions):
    """Configuration options for the standard PDF processing pipeline.

    This class extends `PaginatedPipelineOptions` with a comprehensive set of
    settings for processing PDF documents. It allows for fine-grained control
    over features like table structure recognition, OCR, code and formula
    enrichment, and layout analysis.

    Attributes:
        do_table_structure: If `True`, enables the detection and extraction of
            table structures.
        do_ocr: If `True`, performs OCR on the document.
        do_code_enrichment: If `True`, enables specialized processing to
            identify and enrich code blocks.
        do_formula_enrichment: If `True`, enables specialized processing to
            identify and extract mathematical formulas in LaTeX format.
        force_backend_text: If `True`, uses text from the backend instead of
            from generative models.
        table_structure_options: A `TableStructureOptions` object for
            configuring table recognition.
        ocr_options: An `OcrOptions` object for configuring the OCR engine.
        layout_options: A `LayoutOptions` object for configuring layout analysis.
        images_scale: The scaling factor for generated images.
        generate_page_images: If `True`, generates an image for each page.
        generate_picture_images: If `True`, generates an image for each picture.
        generate_table_images: A deprecated field; use `generate_page_images`
            and `TableItem.get_image` instead.
        generate_parsed_pages: If `True`, includes the raw parsed page data in
            the output.
    """

    do_table_structure: bool = True  # True: perform table structure extraction
    do_ocr: bool = True  # True: perform OCR, replace programmatic PDF text
    do_code_enrichment: bool = False  # True: perform code OCR
    do_formula_enrichment: bool = False  # True: perform formula OCR, return Latex code
    force_backend_text: bool = (
        False  # (To be used with vlms, or other generative models)
    )
    # If True, text from backend will be used instead of generated text

    table_structure_options: TableStructureOptions = TableStructureOptions()
    ocr_options: OcrOptions = EasyOcrOptions()
    layout_options: LayoutOptions = LayoutOptions()

    images_scale: float = 1.0
    generate_page_images: bool = False
    generate_picture_images: bool = False
    generate_table_images: bool = Field(
        default=False,
        deprecated=(
            "Field `generate_table_images` is deprecated. "
            "To obtain table images, set `PdfPipelineOptions.generate_page_images = True` "
            "before conversion and then use the `TableItem.get_image` function."
        ),
    )

    generate_parsed_pages: bool = False


class ProcessingPipeline(str, Enum):
    """An enumeration of the available processing pipelines.

    This enum provides a standardized set of identifiers for the different
    end-to-end processing pipelines available in Docling.

    Attributes:
        STANDARD: The standard PDF processing pipeline.
        VLM: The Vision Language Model (VLM) pipeline.
        ASR: The Automatic Speech Recognition (ASR) pipeline for audio files.
    """

    STANDARD = "standard"
    VLM = "vlm"
    ASR = "asr"


class ThreadedPdfPipelineOptions(PdfPipelineOptions):
    """Configuration options for the threaded PDF processing pipeline.

    This class extends `PdfPipelineOptions` with settings for controlling the
    performance of the threaded pipeline, including batch sizes, timeouts, and
    queue sizes for backpressure management.

    Attributes:
        ocr_batch_size: The number of pages to batch together for OCR processing.
        layout_batch_size: The number of pages to batch together for layout analysis.
        table_batch_size: The number of pages to batch together for table recognition.
        batch_timeout_seconds: The maximum time to wait before processing a
            batch, even if it's not full.
        queue_max_size: The maximum number of items to hold in the processing
            queues to manage backpressure.
    """

    # Batch sizes for different stages
    ocr_batch_size: int = 4
    layout_batch_size: int = 4
    table_batch_size: int = 4

    # Timing control
    batch_timeout_seconds: float = 2.0

    # Backpressure and queue control
    queue_max_size: int = 100
