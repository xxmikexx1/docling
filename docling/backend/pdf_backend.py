from abc import ABC, abstractmethod
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import Optional, Set, Union

from docling_core.types.doc import BoundingBox, Size
from docling_core.types.doc.page import SegmentedPdfPage, TextCell
from PIL import Image

from docling.backend.abstract_backend import PaginatedDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument


class PdfPageBackend(ABC):
    """An abstract base class for page-level PDF processing backends.

    This class defines the interface that all page-level PDF backends must
    implement. A page backend is responsible for extracting content from a
    single page of a PDF document.
    """

    @abstractmethod
    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        """Extracts text from a given rectangular area of the page."""
        pass

    @abstractmethod
    def get_segmented_page(self) -> Optional[SegmentedPdfPage]:
        """Returns a structured representation of the page's content."""
        pass

    @abstractmethod
    def get_text_cells(self) -> Iterable[TextCell]:
        """Returns an iterable of all text cells on the page."""
        pass

    @abstractmethod
    def get_bitmap_rects(self, float: int = 1) -> Iterable[BoundingBox]:
        """Yields the bounding boxes of bitmap images on the page."""
        pass

    @abstractmethod
    def get_page_image(
        self, scale: float = 1, cropbox: Optional[BoundingBox] = None
    ) -> Image.Image:
        """Renders an image of the page."""
        pass

    @abstractmethod
    def get_size(self) -> Size:
        """Returns the size of the page in points."""
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        """Checks if the page backend was initialized successfully."""
        pass

    @abstractmethod
    def unload(self):
        """Releases any resources held by the page backend."""
        pass


class PdfDocumentBackend(PaginatedDocumentBackend):
    """An abstract base class for document-level PDF processing backends.

    This class defines the interface for backends that process entire PDF
    documents. It handles the case where the input is an image, converting it
    to a temporary PDF to be processed by the concrete backend implementation.
    """

    def __init__(self, in_doc: InputDocument, path_or_stream: Union[BytesIO, Path]):
        """Initializes the PdfDocumentBackend.

        This checks if the input format is PDF or an image. If it's an image,
        it converts it to a single-page or multi-page PDF in memory.

        Args:
            in_doc: The `InputDocument` object.
            path_or_stream: The path or stream of the source file.

        Raises:
            RuntimeError: If the input format is not PDF or IMAGE.
        """
        super().__init__(in_doc, path_or_stream)

        if self.input_format is not InputFormat.PDF:
            if self.input_format is InputFormat.IMAGE:
                buf = BytesIO()
                img = Image.open(self.path_or_stream)

                # Handle multi-page TIFF images
                if hasattr(img, "n_frames") and img.n_frames > 1:
                    # Extract all frames from multi-page image
                    frames = []
                    try:
                        for i in range(img.n_frames):
                            img.seek(i)
                            frame = img.copy().convert("RGB")
                            frames.append(frame)
                    except EOFError:
                        pass

                    # Save as multi-page PDF
                    if frames:
                        frames[0].save(
                            buf, "PDF", save_all=True, append_images=frames[1:]
                        )
                    else:
                        # Fallback to single page if frame extraction fails
                        img.convert("RGB").save(buf, "PDF")
                else:
                    # Single page image - convert to RGB and save
                    img.convert("RGB").save(buf, "PDF")

                buf.seek(0)
                self.path_or_stream = buf
            elif self.input_format not in self.supported_formats():
                raise RuntimeError(
                    f"Incompatible file format {self.input_format} was passed to a PdfDocumentBackend. Valid format are {','.join(self.supported_formats())}."
                )

    @abstractmethod
    def load_page(self, page_no: int) -> PdfPageBackend:
        """Loads a single page and returns a `PdfPageBackend` for it."""
        pass

    @abstractmethod
    def page_count(self) -> int:
        """Returns the total number of pages in the document."""
        pass

    @classmethod
    def supported_formats(cls) -> Set[InputFormat]:
        """Returns the set of supported formats, which includes PDF and IMAGE."""
        return {InputFormat.PDF, InputFormat.IMAGE}

    @classmethod
    def supports_pagination(cls) -> bool:
        """PDF backends support pagination."""
        return True
