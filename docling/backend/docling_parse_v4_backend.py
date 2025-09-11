import logging
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import pypdfium2 as pdfium
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import SegmentedPdfPage, TextCell
from docling_parse.pdf_parser import DoclingPdfParser, PdfDocument
from PIL import Image
from pypdfium2 import PdfPage

from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.base_models import Size
from docling.utils.locks import pypdfium2_lock

if TYPE_CHECKING:
    from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


class DoclingParseV4PageBackend(PdfPageBackend):
    """A page-level backend that uses `docling-parse` v4 to process a single PDF page.

    This class handles the extraction of content from a single page of a PDF file,
    leveraging the v4 `docling-parse` parser. It performs lazy parsing of the
    page content to improve efficiency.

    Attributes:
        valid: A boolean indicating whether the page backend was initialized
            successfully.
    """

    def __init__(
        self,
        *,
        dp_doc: PdfDocument,
        page_obj: PdfPage,
        page_no: int,
        create_words: bool = True,
        create_textlines: bool = True,
        keep_chars: bool = False,
        keep_lines: bool = False,
        keep_images: bool = True,
    ):
        """Initializes the DoclingParseV4PageBackend.

        Args:
            dp_doc: The `DoclingPdfParser` document object.
            page_obj: The `pypdfium2.PdfPage` object for this page.
            page_no: The page number (0-indexed).
            create_words: If `True`, generates word-level text cells.
            create_textlines: If `True`, generates textline-level cells.
            keep_chars: If `True`, keeps character-level information.
            keep_lines: If `True`, keeps geometric line information.
            keep_images: If `True`, keeps image information.
        """
        self._ppage = page_obj
        self._dp_doc = dp_doc
        self._page_no = page_no

        self._create_words = create_words
        self._create_textlines = create_textlines

        self._keep_chars = keep_chars
        self._keep_lines = keep_lines
        self._keep_images = keep_images

        self._dpage: Optional[SegmentedPdfPage] = None
        self._unloaded = False
        self.valid = (self._ppage is not None) and (self._dp_doc is not None)

    def _ensure_parsed(self) -> None:
        if self._dpage is not None:
            return

        seg_page = self._dp_doc.get_page(
            self._page_no + 1,
            keep_chars=self._keep_chars,
            keep_lines=self._keep_lines,
            keep_bitmaps=self._keep_images,
            create_words=self._create_words,
            create_textlines=self._create_textlines,
            enforce_same_font=True,
        )

        # In Docling, all TextCell instances are expected with top-left origin.
        [
            tc.to_top_left_origin(seg_page.dimension.height)
            for tc in seg_page.textline_cells
        ]
        [tc.to_top_left_origin(seg_page.dimension.height) for tc in seg_page.char_cells]
        [tc.to_top_left_origin(seg_page.dimension.height) for tc in seg_page.word_cells]

        self._dpage = seg_page

    def is_valid(self) -> bool:
        """Checks if the page backend was initialized successfully."""
        return self.valid

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        """Extracts text from a given rectangular area of the page.

        Args:
            bbox: The `BoundingBox` defining the area to extract text from.

        Returns:
            A string containing the concatenated text of all cells that
            significantly overlap with the bounding box.
        """
        self._ensure_parsed()
        assert self._dpage is not None

        # Find intersecting cells on the page
        text_piece = ""
        page_size = self.get_size()

        scale = (
            1  # FIX - Replace with param in get_text_in_rect across backends (optional)
        )

        for i, cell in enumerate(self._dpage.textline_cells):
            cell_bbox = (
                cell.rect.to_bounding_box()
                .to_top_left_origin(page_height=page_size.height)
                .scaled(scale)
            )

            overlap_frac = cell_bbox.intersection_over_self(bbox)

            if overlap_frac > 0.5:
                if len(text_piece) > 0:
                    text_piece += " "
                text_piece += cell.text

        return text_piece

    def get_segmented_page(self) -> Optional[SegmentedPdfPage]:
        """Returns the `SegmentedPdfPage` object for this page.

        This method triggers the actual parsing of the page if it has not been
        done already.

        Returns:
            The `SegmentedPdfPage` object, or `None` if parsing fails.
        """
        self._ensure_parsed()
        return self._dpage

    def get_text_cells(self) -> Iterable[TextCell]:
        """Returns an iterable of all text cells on the page."""
        self._ensure_parsed()
        assert self._dpage is not None

        return self._dpage.textline_cells

    def get_bitmap_rects(self, scale: float = 1) -> Iterable[BoundingBox]:
        """Yields the bounding boxes of bitmap images on the page.

        Args:
            scale: A scaling factor to apply to the bounding box coordinates.

        Yields:
            A `BoundingBox` for each bitmap image on the page.
        """
        self._ensure_parsed()
        assert self._dpage is not None

        AREA_THRESHOLD = 0  # 32 * 32

        images = self._dpage.bitmap_resources

        for img in images:
            cropbox = img.rect.to_bounding_box().to_top_left_origin(
                self.get_size().height
            )

            if cropbox.area() > AREA_THRESHOLD:
                cropbox = cropbox.scaled(scale=scale)

                yield cropbox

    def get_page_image(
        self, scale: float = 1, cropbox: Optional[BoundingBox] = None
    ) -> Image.Image:
        """Renders an image of the page.

        Args:
            scale: The scaling factor for the rendered image.
            cropbox: An optional `BoundingBox` to crop the image to.

        Returns:
            A `PIL.Image.Image` object of the page.
        """
        page_size = self.get_size()

        if not cropbox:
            cropbox = BoundingBox(
                l=0,
                r=page_size.width,
                t=0,
                b=page_size.height,
                coord_origin=CoordOrigin.TOPLEFT,
            )
            padbox = BoundingBox(
                l=0, r=0, t=0, b=0, coord_origin=CoordOrigin.BOTTOMLEFT
            )
        else:
            padbox = cropbox.to_bottom_left_origin(page_size.height).model_copy()
            padbox.r = page_size.width - padbox.r
            padbox.t = page_size.height - padbox.t

        with pypdfium2_lock:
            image = (
                self._ppage.render(
                    scale=scale * 1.5,
                    rotation=0,  # no additional rotation
                    crop=padbox.as_tuple(),
                )
                .to_pil()
                .resize(
                    size=(round(cropbox.width * scale), round(cropbox.height * scale))
                )
            )  # We resize the image from 1.5x the given scale to make it sharper.

        return image

    def get_size(self) -> Size:
        """Returns the size of the page in points."""
        with pypdfium2_lock:
            return Size(width=self._ppage.get_width(), height=self._ppage.get_height())

        # TODO: Take width and height from docling-parse.
        # return Size(
        #    width=self._dpage.dimension.width,
        #    height=self._dpage.dimension.height,
        # )

    def unload(self):
        """Unloads the page from memory to conserve resources."""
        if not self._unloaded and self._dp_doc is not None:
            self._dp_doc.unload_pages((self._page_no + 1, self._page_no + 2))
            self._unloaded = True

        self._ppage = None
        self._dpage = None
        self._dp_doc = None


class DoclingParseV4DocumentBackend(PdfDocumentBackend):
    """A document-level backend that uses `docling-parse` v4 to process a PDF.

    This class orchestrates the processing of a PDF file using the v4 parser
    from the `docling-parse` library. It loads the document and provides a
    method to access individual pages, which are handled by the
    `DoclingParseV4PageBackend`.
    """

    def __init__(self, in_doc: "InputDocument", path_or_stream: Union[BytesIO, Path]):
        """Initializes the DoclingParseV4DocumentBackend.

        Args:
            in_doc: The `InputDocument` object representing the source PDF.
            path_or_stream: The path or stream of the PDF content.

        Raises:
            RuntimeError: If `docling-parse` v4 fails to load the document.
        """
        super().__init__(in_doc, path_or_stream)

        with pypdfium2_lock:
            self._pdoc = pdfium.PdfDocument(self.path_or_stream)
        self.parser = DoclingPdfParser(loglevel="fatal")
        self.dp_doc: PdfDocument = self.parser.load(path_or_stream=self.path_or_stream)
        success = self.dp_doc is not None

        if not success:
            raise RuntimeError(
                f"docling-parse v4 could not load document {self.document_hash}."
            )

    def page_count(self) -> int:
        """Returns the total number of pages in the document."""
        # return len(self._pdoc)  # To be replaced with docling-parse API

        len_1 = len(self._pdoc)
        len_2 = self.dp_doc.number_of_pages()

        if len_1 != len_2:
            _log.error(f"Inconsistent number of pages: {len_1}!={len_2}")

        return len_2

    def load_page(
        self, page_no: int, create_words: bool = True, create_textlines: bool = True
    ) -> DoclingParseV4PageBackend:
        """Loads a single page and returns a `DoclingParseV4PageBackend` for it.

        Args:
            page_no: The page number to load (0-indexed).
            create_words: If `True`, generates word-level text cells.
            create_textlines: If `True`, generates textline-level cells.

        Returns:
            A `DoclingParseV4PageBackend` instance for the specified page.
        """
        with pypdfium2_lock:
            ppage = self._pdoc[page_no]

        return DoclingParseV4PageBackend(
            dp_doc=self.dp_doc,
            page_obj=ppage,
            page_no=page_no,
            create_words=create_words,
            create_textlines=create_textlines,
        )

    def is_valid(self) -> bool:
        """Checks if the document is valid (i.e., has at least one page)."""
        return self.page_count() > 0

    def unload(self):
        """Unloads the document from `docling-parse` and closes the `pypdfium2` document."""
        super().unload()
        # Unload docling-parse document first
        if self.dp_doc is not None:
            self.dp_doc.unload()
            self.dp_doc = None

        # Then close pypdfium2 document with proper locking
        if self._pdoc is not None:
            with pypdfium2_lock:
                try:
                    self._pdoc.close()
                except Exception:
                    # Ignore cleanup errors
                    pass
            self._pdoc = None
