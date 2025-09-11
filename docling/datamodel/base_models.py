import math
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Type, Union

import numpy as np
from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    NodeItem,
    PictureDataType,
    Size,
    TableCell,
)
from docling_core.types.doc.base import PydanticSerCtxKey, round_pydantic_float
from docling_core.types.doc.page import SegmentedPdfPage, TextCell
from docling_core.types.io import (
    DocumentStream,
)

# DO NOT REMOVE; explicitly exposed from this location
from PIL.Image import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FieldSerializationInfo,
    computed_field,
    field_serializer,
)

if TYPE_CHECKING:
    from docling.backend.pdf_backend import PdfPageBackend

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.datamodel.pipeline_options import PipelineOptions


class BaseFormatOption(BaseModel):
    """A base model for defining format-specific conversion options.

    This class serves as a foundation for creating configuration objects that
    specify how a particular document format should be processed. It includes
    common options like pipeline settings and the backend responsible for handling
    the format.

    Attributes:
        pipeline_options: An optional `PipelineOptions` object that contains
            detailed settings for the document processing pipeline.
        backend: The `AbstractDocumentBackend` class responsible for parsing
            and converting the document format.
    """

    pipeline_options: Optional[PipelineOptions] = None
    backend: Type[AbstractDocumentBackend]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ConversionStatus(str, Enum):
    """An enumeration of the possible statuses of a document conversion process.

    This enum provides a standardized set of states to track the progress and
    outcome of a conversion task, from initiation to completion or failure.

    Attributes:
        PENDING: The conversion has been scheduled but has not yet started.
        STARTED: The conversion process is currently active.
        FAILURE: The conversion failed to complete due to an error.
        SUCCESS: The conversion completed successfully without any issues.
        PARTIAL_SUCCESS: The conversion completed, but some parts of the
            document may have been processed incorrectly or omitted.
        SKIPPED: The conversion was skipped, typically because the file
            format is not supported or was excluded by user settings.
    """

    PENDING = "pending"
    STARTED = "started"
    FAILURE = "failure"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    SKIPPED = "skipped"


class InputFormat(str, Enum):
    """An enumeration of input document formats supported by backend parsers.

    This enum defines the set of document formats that Docling can parse and
    process. Each member corresponds to a specific file type and is used to
    select the appropriate backend for conversion.

    Attributes:
        DOCX: Microsoft Word document format (.docx).
        PPTX: Microsoft PowerPoint presentation format (.pptx).
        HTML: HyperText Markup Language format (.html, .htm).
        IMAGE: Various image formats (e.g., .png, .jpg, .tiff).
        PDF: Portable Document Format (.pdf).
        ASCIIDOC: AsciiDoc lightweight markup language format (.asciidoc, .adoc).
        MD: Markdown format (.md).
        CSV: Comma-Separated Values format (.csv).
        XLSX: Microsoft Excel spreadsheet format (.xlsx).
        XML_USPTO: United States Patent and Trademark Office XML format.
        XML_JATS: Journal Article Tag Suite (JATS) XML format.
        METS_GBS: Metadata Encoding and Transmission Standard (METS) for Google Books.
        JSON_DOCLING: Docling's native JSON format for representing documents.
        AUDIO: Various audio formats (e.g., .wav, .mp3).
    """

    DOCX = "docx"
    PPTX = "pptx"
    HTML = "html"
    IMAGE = "image"
    PDF = "pdf"
    ASCIIDOC = "asciidoc"
    MD = "md"
    CSV = "csv"
    XLSX = "xlsx"
    XML_USPTO = "xml_uspto"
    XML_JATS = "xml_jats"
    METS_GBS = "mets_gbs"
    JSON_DOCLING = "json_docling"
    AUDIO = "audio"


class OutputFormat(str, Enum):
    """An enumeration of the supported output formats for document conversion.

    This enum defines the various formats to which a processed document can be
    exported.

    Attributes:
        MARKDOWN: Exports the document to Markdown format (.md).
        JSON: Exports the document to Docling's native JSON format.
        HTML: Exports the document to a single HTML file.
        HTML_SPLIT_PAGE: Exports the document to multiple HTML files, with one
            file per page.
        TEXT: Exports the document's plain text content.
        DOCTAGS: Exports the document using DocTags, a structured tagging format.
    """

    MARKDOWN = "md"
    JSON = "json"
    HTML = "html"
    HTML_SPLIT_PAGE = "html_split_page"
    TEXT = "text"
    DOCTAGS = "doctags"


FormatToExtensions: Dict[InputFormat, List[str]] = {
    InputFormat.DOCX: ["docx", "dotx", "docm", "dotm"],
    InputFormat.PPTX: ["pptx", "potx", "ppsx", "pptm", "potm", "ppsm"],
    InputFormat.PDF: ["pdf"],
    InputFormat.MD: ["md"],
    InputFormat.HTML: ["html", "htm", "xhtml"],
    InputFormat.XML_JATS: ["xml", "nxml"],
    InputFormat.IMAGE: ["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"],
    InputFormat.ASCIIDOC: ["adoc", "asciidoc", "asc"],
    InputFormat.CSV: ["csv"],
    InputFormat.XLSX: ["xlsx", "xlsm"],
    InputFormat.XML_USPTO: ["xml", "txt"],
    InputFormat.METS_GBS: ["tar.gz"],
    InputFormat.JSON_DOCLING: ["json"],
    InputFormat.AUDIO: ["wav", "mp3"],
}

FormatToMimeType: Dict[InputFormat, List[str]] = {
    InputFormat.DOCX: [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.template",
    ],
    InputFormat.PPTX: [
        "application/vnd.openxmlformats-officedocument.presentationml.template",
        "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ],
    InputFormat.HTML: ["text/html", "application/xhtml+xml"],
    InputFormat.XML_JATS: ["application/xml"],
    InputFormat.IMAGE: [
        "image/png",
        "image/jpeg",
        "image/tiff",
        "image/gif",
        "image/bmp",
        "image/webp",
    ],
    InputFormat.PDF: ["application/pdf"],
    InputFormat.ASCIIDOC: ["text/asciidoc"],
    InputFormat.MD: ["text/markdown", "text/x-markdown"],
    InputFormat.CSV: ["text/csv"],
    InputFormat.XLSX: [
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ],
    InputFormat.XML_USPTO: ["application/xml", "text/plain"],
    InputFormat.METS_GBS: ["application/mets+xml"],
    InputFormat.JSON_DOCLING: ["application/json"],
    InputFormat.AUDIO: ["audio/x-wav", "audio/mpeg", "audio/wav", "audio/mp3"],
}

MimeTypeToFormat: dict[str, list[InputFormat]] = {
    mime: [fmt for fmt in FormatToMimeType if mime in FormatToMimeType[fmt]]
    for value in FormatToMimeType.values()
    for mime in value
}


class DocInputType(str, Enum):
    """An enumeration for the type of document input source.

    This enum distinguishes between different ways a document can be provided
    as input: either as a file path on the local system or as a data stream.

    Attributes:
        PATH: The document is specified by a local file path.
        STREAM: The document is provided as an in-memory data stream.
    """

    PATH = "path"
    STREAM = "stream"


class DoclingComponentType(str, Enum):
    """An enumeration for the different types of components within the Docling system.

    This enum is used to identify the source of events, errors, or logs within
    the Docling architecture, making it easier to trace and debug issues.

    Attributes:
        DOCUMENT_BACKEND: A component responsible for parsing and converting a
            specific document format.
        MODEL: A machine learning model used for tasks like layout analysis,
            OCR, or classification.
        DOC_ASSEMBLER: The component that assembles parsed data and model
            predictions into a coherent document structure.
        USER_INPUT: Represents an error or issue originating from user-provided
            input, such as an invalid file or configuration.
    """

    DOCUMENT_BACKEND = "document_backend"
    MODEL = "model"
    DOC_ASSEMBLER = "doc_assembler"
    USER_INPUT = "user_input"


class ErrorItem(BaseModel):
    """A data model for representing an error that occurred within Docling.

    This class encapsulates the details of an error, including its source component,
    the module where it occurred, and a descriptive message. It provides a
    structured way to report and log errors throughout the system.

    Attributes:
        component_type: The `DoclingComponentType` that generated the error.
        module_name: The name of the module where the error was raised.
        error_message: A string containing the details of the error.
    """

    component_type: DoclingComponentType
    module_name: str
    error_message: str


class Cluster(BaseModel):
    """Represents a cluster of text or other content within a document page.

    A cluster is a fundamental unit for grouping related content, such as a
    paragraph, a table cell, or a list item. It includes a unique identifier,
    a label describing its type, a bounding box defining its location, and its
    constituent elements.

    Attributes:
        id: A unique integer identifier for the cluster.
        label: A `DocItemLabel` that categorizes the cluster's content
            (e.g., "paragraph", "table").
        bbox: A `BoundingBox` object that specifies the coordinates of the
            cluster on the page.
        confidence: The confidence score of the cluster prediction, ranging
            from 0.0 to 1.0. Defaults to 1.0.
        cells: A list of `TextCell` objects that make up the content of the
            cluster.
        children: A list of nested `Cluster` objects, allowing for hierarchical
            content structures.
    """

    id: int
    label: DocItemLabel
    bbox: BoundingBox
    confidence: float = 1.0
    cells: List[TextCell] = []
    children: List["Cluster"] = []  # Add child cluster support

    @field_serializer("confidence")
    def _serialize(self, value: float, info: FieldSerializationInfo) -> float:
        return round_pydantic_float(value, info.context, PydanticSerCtxKey.CONFID_PREC)


class BasePageElement(BaseModel):
    """A base model for elements that belong to a specific page in a document.

    This class provides a common structure for all page-level elements, such as
    text blocks, tables, and figures. It includes essential attributes like an
    identifier, a label, the page number, and the underlying cluster of content.

    Attributes:
        label: A `DocItemLabel` that categorizes the element's content type.
        id: A unique integer identifier for the element.
        page_no: The page number where the element is located.
        cluster: The `Cluster` object that contains the raw content and
            geometrical information for this element.
        text: The textual content of the element, if applicable.
    """

    label: DocItemLabel
    id: int
    page_no: int
    cluster: Cluster
    text: Optional[str] = None


class LayoutPrediction(BaseModel):
    """Represents the output of a layout analysis model for a single page.

    This class holds the results of layout detection, which consists of a list
    of `Cluster` objects that identify and group the various content elements
    on a page.

    Attributes:
        clusters: A list of `Cluster` objects, where each cluster represents a
            distinct block of content (e.g., a paragraph, heading, or table).
    """

    clusters: List[Cluster] = []


class VlmPredictionToken(BaseModel):
    """Represents a single token generated by a Vision Language Model (VLM).

    This class encapsulates the information associated with a single token in a
    sequence generated by a VLM, including its text representation, numerical
    token ID, and log probability.

    Attributes:
        text: The string representation of the token.
        token: The numerical identifier of the token in the model's vocabulary.
        logprob: The log probability of the token, indicating the model's
            confidence in its prediction.
    """

    text: str = ""
    token: int = -1
    logprob: float = -1


class VlmPrediction(BaseModel):
    """Represents the complete output of a Vision Language Model (VLM) prediction.

    This class aggregates the results of a VLM inference, including the full
    generated text, a list of individual tokens with their metadata, and the
    total generation time.

    Attributes:
        text: The complete string of text generated by the VLM.
        generated_tokens: A list of `VlmPredictionToken` objects, providing
            detailed information about each token in the generated sequence.
        generation_time: The time in seconds it took for the model to generate
            the prediction.
    """

    text: str = ""
    generated_tokens: list[VlmPredictionToken] = []
    generation_time: float = -1


class ContainerElement(BasePageElement):
    """A specialized page element used for grouping other elements.

    This class is primarily used for typing purposes to represent container-like
    structures such as forms or key-value regions, which logically group other
    page elements together. It inherits from `BasePageElement` and does not add
    any new attributes.
    """

    pass


class Table(BasePageElement):
    """Represents a table extracted from a document page.

    This class models a table, including its structure, dimensions, and the
    content of its cells. It inherits from `BasePageElement` and adds
    table-specific attributes.

    Attributes:
        otsl_seq: A list of strings representing the table's structure in
            Open-source Table Structure Linter (OTSL) format.
        num_rows: The number of rows in the table.
        num_cols: The number of columns in the table.
        table_cells: A list of `TableCell` objects, each containing the content
            and properties of a single cell in the table.
    """

    otsl_seq: List[str]
    num_rows: int = 0
    num_cols: int = 0
    table_cells: List[TableCell]


class TableStructurePrediction(BaseModel):
    """Represents the output of a table structure recognition model.

    This class holds the results of table detection for a single page, mapping
    table identifiers to their corresponding `Table` objects.

    Attributes:
        table_map: A dictionary where keys are integer identifiers for tables
            and values are the `Table` objects containing the structured data.
    """

    table_map: Dict[int, Table] = {}


class TextElement(BasePageElement):
    """Represents a block of text extracted from a document page.

    This class is a specialized `BasePageElement` for handling textual content.
    It includes a `text` attribute that holds the string content of the element.

    Attributes:
        text: The string content of the text element.
    """

    text: str


class FigureElement(BasePageElement):
    """Represents a figure or image extracted from a document page.

    This class models a visual element, such as a chart, photograph, or diagram.
    It includes attributes for storing annotations, provenance information, and
    classification results.

    Attributes:
        annotations: A list of `PictureDataType` objects that provide detailed
            annotations for the figure (e.g., identifying sub-components).
        provenance: An optional string indicating the origin or source of the
            figure.
        predicted_class: The classification label predicted for the figure
            (e.g., "bar_chart", "photograph").
        confidence: The confidence score of the classification prediction.
    """

    annotations: List[PictureDataType] = []
    provenance: Optional[str] = None
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None

    @field_serializer("confidence")
    def _serialize(
        self, value: Optional[float], info: FieldSerializationInfo
    ) -> Optional[float]:
        return (
            round_pydantic_float(value, info.context, PydanticSerCtxKey.CONFID_PREC)
            if value is not None
            else None
        )


class FigureClassificationPrediction(BaseModel):
    """Represents the output of a figure classification model.

    This class holds the results of figure detection and classification for a
    single page. It includes a count of the detected figures and a map from
    figure identifiers to their corresponding `FigureElement` objects.

    Attributes:
        figure_count: The total number of figures detected on the page.
        figure_map: A dictionary where keys are integer identifiers for figures
            and values are the `FigureElement` objects containing the figure's
            details.
    """

    figure_count: int = 0
    figure_map: Dict[int, FigureElement] = {}


class EquationPrediction(BaseModel):
    """Represents the output of an equation detection model.

    This class holds the results of equation detection for a single page. It
    includes a count of the detected equations and a map from equation
    identifiers to their corresponding `TextElement` objects.

    Attributes:
        equation_count: The total number of equations detected on the page.
        equation_map: A dictionary where keys are integer identifiers for
            equations and values are the `TextElement` objects containing the
            equation's content.
    """

    equation_count: int = 0
    equation_map: Dict[int, TextElement] = {}


class PagePredictions(BaseModel):
    """Aggregates all model predictions for a single document page.

    This class serves as a container for the outputs of various models that may
    be run on a page, such as layout analysis, table structure recognition,
    figure classification, and equation detection.

    Attributes:
        layout: An optional `LayoutPrediction` object containing the results of
            layout analysis.
        tablestructure: An optional `TableStructurePrediction` object with
            recognized table structures.
        figures_classification: An optional `FigureClassificationPrediction`
            object with classified figures.
        equations_prediction: An optional `EquationPrediction` object with
            detected equations.
        vlm_response: An optional `VlmPrediction` object containing the output
            from a Vision Language Model.
    """

    layout: Optional[LayoutPrediction] = None
    tablestructure: Optional[TableStructurePrediction] = None
    figures_classification: Optional[FigureClassificationPrediction] = None
    equations_prediction: Optional[EquationPrediction] = None
    vlm_response: Optional[VlmPrediction] = None


PageElement = Union[TextElement, Table, FigureElement, ContainerElement]


class AssembledUnit(BaseModel):
    """Represents the assembled content of a page, organized into logical sections.

    This class holds the final, structured content of a page after it has been
    processed and assembled. It separates elements into body and header sections,
    providing a clean representation of the page's structure.

    Attributes:
        elements: A list of all `PageElement` objects on the page, in their
            original order.
        body: A list of `PageElement` objects that constitute the main content
            of the page.
        headers: A list of `PageElement` objects that are part of the page's
            header.
    """

    elements: List[PageElement] = []
    body: List[PageElement] = []
    headers: List[PageElement] = []


class ItemAndImageEnrichmentElement(BaseModel):
    """A data structure for holding an item and its corresponding image for enrichment.

    This class is used to package a `NodeItem` with its associated `Image` object,
    typically for tasks that involve enriching the item with information derived
    from the image, such as visual analysis or OCR.

    Attributes:
        item: The `NodeItem` to be enriched.
        image: The `Image` object associated with the item.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    item: NodeItem
    image: Image


class Page(BaseModel):
    """Represents a single page within a document.

    This class is a central data structure that encapsulates all information
    related to a single page, including its number, dimensions, parsed content,
    model predictions, and assembled structure. It also provides methods for
    accessing page images and other derived data.

    Attributes:
        page_no: The page number (1-based).
        size: An optional `Size` object specifying the dimensions of the page.
        parsed_page: An optional `SegmentedPdfPage` containing the raw parsed
            content from the backend.
        predictions: A `PagePredictions` object that aggregates all model
            predictions for the page.
        assembled: An optional `AssembledUnit` that holds the final, structured
            content of the page.
        cells: A read-only property that returns a list of `TextCell` objects
            from the parsed page.
        image: A read-only property that returns the page's image at the
            default scale.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    page_no: int
    # page_hash: Optional[str] = None
    size: Optional[Size] = None
    parsed_page: Optional[SegmentedPdfPage] = None
    predictions: PagePredictions = PagePredictions()
    assembled: Optional[AssembledUnit] = None

    _backend: Optional["PdfPageBackend"] = (
        None  # Internal PDF backend. By default it is cleared during assembling.
    )
    _default_image_scale: float = 1.0  # Default image scale for external usage.
    _image_cache: Dict[
        float, Image
    ] = {}  # Cache of images in different scales. By default it is cleared during assembling.

    @property
    def cells(self) -> List[TextCell]:
        """Provides a read-only view of the text cells on the page.

        This property returns a list of `TextCell` objects, which represent the
        individual text fragments extracted from the page during the parsing stage.

        Returns:
            A list of `TextCell` objects. If the page has not been parsed, an
            empty list is returned.
        """
        if self.parsed_page is not None:
            return self.parsed_page.textline_cells
        else:
            return []

    def get_image(
        self,
        scale: float = 1.0,
        max_size: Optional[int] = None,
        cropbox: Optional[BoundingBox] = None,
    ) -> Optional[Image]:
        """Retrieves an image of the page with specified scaling and cropping.

        This method generates or retrieves a cached image of the page. It allows
        for resizing the image by a scale factor or to a maximum dimension, and
        for cropping to a specific bounding box.

        Args:
            scale: The scaling factor to apply to the image. Defaults to 1.0.
            max_size: An optional integer specifying the maximum size (width or
                height) of the output image. The scale is adjusted to fit within
                this size.
            cropbox: An optional `BoundingBox` to crop the image to.

        Returns:
            An optional `Image` object representing the page's visual content.
            Returns `None` if the page backend is not available.
        """
        if self._backend is None:
            return self._image_cache.get(scale, None)

        if max_size:
            assert self.size is not None
            scale = min(scale, max_size / max(self.size.as_tuple()))

        if scale not in self._image_cache:
            if cropbox is None:
                self._image_cache[scale] = self._backend.get_page_image(scale=scale)
            else:
                return self._backend.get_page_image(scale=scale, cropbox=cropbox)

        if cropbox is None:
            return self._image_cache[scale]
        else:
            page_im = self._image_cache[scale]
            assert self.size is not None
            return page_im.crop(
                cropbox.to_top_left_origin(page_height=self.size.height)
                .scaled(scale=scale)
                .as_tuple()
            )

    @property
    def image(self) -> Optional[Image]:
        """Provides the default image of the page.

        This property is a convenient shortcut for `get_image()` with the default
        scale. It returns the visual representation of the page as an `Image` object.

        Returns:
            An optional `Image` object of the page.
        """
        return self.get_image(scale=self._default_image_scale)


## OpenAI API Request / Response Models ##


class OpenAiChatMessage(BaseModel):
    """Represents a single message in a chat conversation, following the OpenAI API format.

    This class models a message within a chat, specifying the role of the
    speaker (e.g., "user", "assistant") and the content of the message.

    Attributes:
        role: The role of the message's author (e.g., "system", "user",
            "assistant").
        content: The text content of the message.
    """

    role: str
    content: str


class OpenAiResponseChoice(BaseModel):
    """Represents a single choice in an OpenAI API response.

    When the OpenAI API returns multiple possible completions, each one is
    encapsulated in a "choice" object. This class models that structure,
    including the message content and the reason the model finished generating.

    Attributes:
        index: The index of this choice in the list of choices.
        message: The `OpenAiChatMessage` containing the generated content.
        finish_reason: The reason the model stopped generating tokens, such as
            "stop" (reached a stop sequence) or "length" (reached the maximum
            token limit).
    """

    index: int
    message: OpenAiChatMessage
    finish_reason: Optional[str]


class OpenAiResponseUsage(BaseModel):
    """Represents the token usage statistics for an OpenAI API request.

    This class provides a breakdown of how many tokens were used in the prompt
    and the completion, as well as the total number of tokens consumed by the API
    call.

    Attributes:
        prompt_tokens: The number of tokens in the input prompt.
        completion_tokens: The number of tokens in the generated completion.
        total_tokens: The total number of tokens used in the request.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAiApiResponse(BaseModel):
    """Represents the full response from an OpenAI API chat completion request.

    This class models the top-level structure of a response from the OpenAI API,
    including the request ID, the model used, the list of choices, creation
    timestamp, and token usage statistics.

    Attributes:
        id: A unique identifier for the API request.
        model: The name of the model that generated the response.
        choices: A list of `OpenAiResponseChoice` objects, each representing a
            possible completion.
        created: The Unix timestamp of when the response was created.
        usage: An `OpenAiResponseUsage` object detailing the token consumption.
    """

    model_config = ConfigDict(
        protected_namespaces=(),
    )

    id: str
    model: Optional[str] = None  # returned by openai
    choices: List[OpenAiResponseChoice]
    created: int
    usage: OpenAiResponseUsage


# Create a type alias for score values
ScoreValue = float


class QualityGrade(str, Enum):
    """An enumeration for grading the quality of document processing results.

    This enum provides a qualitative scale to assess the quality of various
    processing outputs, such as parsing, layout analysis, and OCR. The grades
    are derived from numerical confidence scores.

    Attributes:
        POOR: The processing quality is low, likely containing significant errors.
        FAIR: The processing quality is acceptable but may have some inaccuracies.
        GOOD: The processing quality is high, with few or no errors.
        EXCELLENT: The processing quality is outstanding, with very high confidence.
        UNSPECIFIED: The quality grade could not be determined.
    """

    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"
    UNSPECIFIED = "unspecified"


class PageConfidenceScores(BaseModel):
    """Represents the confidence scores for various processing steps on a single page.

    This class aggregates the confidence scores from different stages of document
    processing, such as parsing, layout analysis, table recognition, and OCR.
    It also provides computed properties to derive qualitative grades from these scores.

    Attributes:
        parse_score: The confidence score for the initial document parsing.
        layout_score: The confidence score for the layout analysis model.
        table_score: The confidence score for table structure recognition.
        ocr_score: The confidence score for Optical Character Recognition (OCR).
        mean_grade: A computed property that returns the `QualityGrade` based on the
            mean of all scores.
        low_grade: A computed property that returns the `QualityGrade` based on the
            5th percentile of all scores, representing a lower-bound estimate.
        mean_score: A computed property that calculates the mean of all scores.
        low_score: A computed property that calculates the 5th percentile of all scores.
    """

    parse_score: ScoreValue = np.nan
    layout_score: ScoreValue = np.nan
    table_score: ScoreValue = np.nan
    ocr_score: ScoreValue = np.nan

    def _score_to_grade(self, score: ScoreValue) -> QualityGrade:
        """Converts a numerical score to a qualitative `QualityGrade`.

        Args:
            score: The numerical confidence score, typically between 0 and 1.

        Returns:
            The corresponding `QualityGrade`.
        """
        if score < 0.5:
            return QualityGrade.POOR
        elif score < 0.8:
            return QualityGrade.FAIR
        elif score < 0.9:
            return QualityGrade.GOOD
        elif score >= 0.9:
            return QualityGrade.EXCELLENT

        return QualityGrade.UNSPECIFIED

    @computed_field  # type: ignore
    @property
    def mean_grade(self) -> QualityGrade:
        """The quality grade based on the mean confidence score."""
        return self._score_to_grade(self.mean_score)

    @computed_field  # type: ignore
    @property
    def low_grade(self) -> QualityGrade:
        """The quality grade based on the low-end confidence score (5th percentile)."""
        return self._score_to_grade(self.low_score)

    @computed_field  # type: ignore
    @property
    def mean_score(self) -> ScoreValue:
        """The mean of all confidence scores for the page."""
        return ScoreValue(
            np.nanmean(
                [
                    self.ocr_score,
                    self.table_score,
                    self.layout_score,
                    self.parse_score,
                ]
            )
        )

    @computed_field  # type: ignore
    @property
    def low_score(self) -> ScoreValue:
        """The 5th percentile of all confidence scores for the page."""
        return ScoreValue(
            np.nanquantile(
                [
                    self.ocr_score,
                    self.table_score,
                    self.layout_score,
                    self.parse_score,
                ],
                q=0.05,
            )
        )


class ConfidenceReport(PageConfidenceScores):
    """Generates a comprehensive confidence report for an entire document.

    This class extends `PageConfidenceScores` to provide an aggregated view of
    confidence scores across all pages of a document. It includes a dictionary
    of per-page scores and computes overall mean and low scores for the entire
    document.

    Attributes:
        pages: A dictionary mapping page numbers to their corresponding
            `PageConfidenceScores` objects.
        mean_score: A computed property that calculates the mean of the mean scores
            of all pages.
        low_score: A computed property that calculates the mean of the low scores
            (5th percentile) of all pages.
    """

    pages: Dict[int, PageConfidenceScores] = Field(
        default_factory=lambda: defaultdict(PageConfidenceScores)
    )

    @computed_field  # type: ignore
    @property
    def mean_score(self) -> ScoreValue:
        """The overall mean confidence score for the entire document."""
        return ScoreValue(
            np.nanmean(
                [c.mean_score for c in self.pages.values()],
            )
        )

    @computed_field  # type: ignore
    @property
    def low_score(self) -> ScoreValue:
        """The overall low-end confidence score for the entire document."""
        return ScoreValue(
            np.nanmean(
                [c.low_score for c in self.pages.values()],
            )
        )
