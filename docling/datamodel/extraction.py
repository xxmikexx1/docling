"""Data models for document extraction functionality."""

from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field

from docling.datamodel.base_models import ConversionStatus, ErrorItem
from docling.datamodel.document import InputDocument


class ExtractedPageData(BaseModel):
    """Represents the extracted data from a single page of a document.

    This class serves as a container for the structured data, raw text, and any
    errors that occurred during the extraction process for a specific page.

    Attributes:
        page_no: The 1-indexed page number from which the data was extracted.
        extracted_data: A dictionary containing the structured data extracted
            from the page, based on the provided template.
        raw_text: The raw text content of the page.
        errors: A list of strings detailing any errors encountered during the
            extraction process for this page.
    """

    page_no: int = Field(..., description="1-indexed page number")
    extracted_data: Optional[Dict[str, Any]] = Field(
        None, description="Extracted structured data from the page"
    )
    raw_text: Optional[str] = Field(None, description="Raw extracted text")
    errors: List[str] = Field(
        default_factory=list,
        description="Any errors encountered during extraction for this page",
    )


class ExtractionResult(BaseModel):
    """Represents the complete result of an extraction process for a document.

    This class aggregates the results from all pages of a document, including the
    overall status, any document-level errors, and a list of `ExtractedPageData`
    objects for each page.

    Attributes:
        input: The `InputDocument` that was processed.
        status: The final `ConversionStatus` of the extraction (e.g., success,
            failure).
        errors: A list of `ErrorItem` objects detailing any document-level
            errors that occurred.
        pages: A list of `ExtractedPageData` objects, each containing the
            extraction results for a single page.
    """

    input: InputDocument
    status: ConversionStatus = ConversionStatus.PENDING
    errors: List[ErrorItem] = []

    # Pages field - always a list for consistency
    pages: List[ExtractedPageData] = Field(
        default_factory=list, description="Extracted data from each page"
    )


# Type alias for template parameters that can be string, dict, or BaseModel
ExtractionTemplateType = Union[str, Dict[str, Any], BaseModel, Type[BaseModel]]
