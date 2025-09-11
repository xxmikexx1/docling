"""Backend for GBS Google Books schema."""

import logging
import tarfile
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

from docling_core.types.doc import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfPageBoundaryType,
    PdfPageGeometry,
    SegmentedPdfPage,
    TextCell,
)
from lxml import etree
from PIL import Image
from PIL.Image import Image as PILImage

from docling.backend.abstract_backend import PaginatedDocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.base_models import InputFormat

if TYPE_CHECKING:
    from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


def _get_pdf_page_geometry(
    size: Size,
) -> PdfPageGeometry:
    boundary_type: PdfPageBoundaryType = PdfPageBoundaryType.CROP_BOX

    bbox_tuple = (0, 0, size.width, size.height)
    bbox = BoundingBox.from_tuple(bbox_tuple, CoordOrigin.TOPLEFT)

    return PdfPageGeometry(
        angle=0.0,
        rect=BoundingRectangle.from_bounding_box(bbox),
        boundary_type=boundary_type,
        art_bbox=bbox,
        bleed_bbox=bbox,
        crop_bbox=bbox,
        media_bbox=bbox,
        trim_bbox=bbox,
    )


class MetsGbsPageBackend(PdfPageBackend):
    """A page-level backend for handling a single page from a METS GBS document.

    This class provides access to the content of a single page, including its
    image and segmented text data.

    Attributes:
        valid: A boolean indicating if the page was loaded successfully.
    """

    def __init__(self, parsed_page: SegmentedPdfPage, page_im: PILImage):
        """Initializes the MetsGbsPageBackend.

        Args:
            parsed_page: The `SegmentedPdfPage` object containing the parsed
                content of the page.
            page_im: The `PIL.Image.Image` object for the page.
        """
        self._im = page_im
        self._dpage = parsed_page
        self.valid = parsed_page is not None

    def is_valid(self) -> bool:
        """Checks if the page was loaded successfully."""
        return self.valid

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        """Extracts text from a given rectangular area of the page.

        Args:
            bbox: The `BoundingBox` defining the area to extract text from.

        Returns:
            A string containing the concatenated text of all cells that
            significantly overlap with the bounding box.
        """
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
        """Returns the `SegmentedPdfPage` object for this page."""
        return self._dpage

    def get_text_cells(self) -> Iterable[TextCell]:
        """Returns an iterable of all text cells on the page."""
        return self._dpage.textline_cells

    def get_bitmap_rects(self, scale: float = 1) -> Iterable[BoundingBox]:
        """Yields the bounding boxes of bitmap images on the page.

        Args:
            scale: A scaling factor to apply to the bounding box coordinates.

        Yields:
            A `BoundingBox` for each bitmap image on the page.
        """
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
        """Returns an image of the page.

        Args:
            scale: The scaling factor for the rendered image.
            cropbox: An optional `BoundingBox` to crop the image to.

        Returns:
            A `PIL.Image.Image` object of the page.
        """
        page_size = self.get_size()
        assert (
            page_size.width == self._im.size[0] and page_size.height == self._im.size[1]
        )

        if not cropbox:
            cropbox = BoundingBox(
                l=0,
                r=page_size.width,
                t=0,
                b=page_size.height,
                coord_origin=CoordOrigin.TOPLEFT,
            )

        image = self._im.resize(
            size=(round(page_size.width * scale), round(page_size.height * scale))
        ).crop(cropbox.scaled(scale=scale).as_tuple())
        return image

    def get_size(self) -> Size:
        """Returns the size of the page in points."""
        return Size(
            width=self._dpage.dimension.width, height=self._dpage.dimension.height
        )

    def unload(self) -> None:
        """Releases the page image and data to free up memory."""
        if hasattr(self, "_im"):
            delattr(self, "_im")
        if hasattr(self, "_dpage"):
            delattr(self, "_dpage")


class _UseType(str, Enum):
    """An enumeration for the `USE` attribute of a `<fileGrp>` element in METS."""

    IMAGE = "image"
    OCR = "OCR"
    COORD_OCR = "coordOCR"


@dataclass
class _FileInfo:
    """A dataclass to store information about a file in the METS archive."""

    file_id: str
    mimetype: str
    path: str
    use: _UseType


@dataclass
class _PageFiles:
    """A dataclass to group the different files associated with a single page."""

    image: Optional[_FileInfo] = None
    ocr: Optional[_FileInfo] = None
    coordOCR: Optional[_FileInfo] = None


def _extract_rect(title_str: str) -> Optional[BoundingRectangle]:
    """Extracts a bounding box from a title string.

    The title string is expected to contain a "bbox" attribute, e.g.,
    'bbox 279 177 306 214;x_wconf 97'.

    Args:
        title_str: The title string to parse.

    Returns:
        A `BoundingRectangle` object, or `None` if parsing fails.
    """
    parts = title_str.split(";")
    for part in parts:
        part = part.strip()
        if part.startswith("bbox "):
            try:
                coords = part.split()[1:]
                rect = BoundingRectangle.from_bounding_box(
                    bbox=BoundingBox.from_tuple(
                        tuple(map(int, coords)), origin=CoordOrigin.TOPLEFT
                    )
                )
                return rect
            except Exception:
                return None
    return None


def _extract_confidence(title_str) -> float:
    """Extracts the OCR confidence (`x_wconf`) from a title string.

    Args:
        title_str: The title string to parse.

    Returns:
        The OCR confidence as a float between 0 and 1, or 1.0 if not found.
    """
    for part in title_str.split(";"):
        part = part.strip()
        if part.startswith("x_wconf"):
            try:
                return float(part.split()[1]) / 100.0
            except Exception:
                return 1
    return 1


class MetsGbsDocumentBackend(PdfDocumentBackend):
    """A document-level backend for handling METS GBS documents.

    This class orchestrates the processing of a METS GBS `tar.gz` archive. It
    parses the main METS XML file to build a map of pages and their associated
    files (images, OCR data), and provides a method to load individual pages.

    Attributes:
        root_mets: The root element of the parsed METS XML file.
        page_map: A dictionary mapping page numbers to `_PageFiles` objects.
    """

    def __init__(self, in_doc: "InputDocument", path_or_stream: Union[BytesIO, Path]):
        """Initializes the MetsGbsDocumentBackend.

        This opens the `tar.gz` archive, finds and parses the main METS XML
        file, and builds a map of all the files in the archive.

        Args:
            in_doc: The `InputDocument` object representing the source archive.
            path_or_stream: The path or stream of the `tar.gz` archive.

        Raises:
            RuntimeError: If a valid METS XML file cannot be found in the archive.
        """
        super().__init__(in_doc, path_or_stream)

        self._tar: tarfile.TarFile = (
            tarfile.open(name=self.path_or_stream, mode="r:gz")
            if isinstance(self.path_or_stream, Path)
            else tarfile.open(fileobj=self.path_or_stream, mode="r:gz")
        )
        self.root_mets: Optional[etree._Element] = None
        self.page_map: Dict[int, _PageFiles] = {}

        for member in self._tar.getmembers():
            if member.name.endswith(".xml"):
                file = self._tar.extractfile(member)
                if file is not None:
                    content = file.read()
                    self.root_mets = self._validate_mets_xml(content)
                    if self.root_mets is not None:
                        break

        if self.root_mets is None:
            raise RuntimeError(
                f"METS GBS backend could not load document {self.document_hash}."
            )

        ns = {
            "mets": "http://www.loc.gov/METS/",
            "xlink": "http://www.w3.org/1999/xlink",
            "xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "gbs": "http://books.google.com/gbs",
            "premis": "info:lc/xmlns/premis-v2",
            "marc": "http://www.loc.gov/MARC21/slim",
        }

        file_info_by_id: Dict[str, _FileInfo] = {}

        for filegrp in self.root_mets.xpath(".//mets:fileGrp", namespaces=ns):
            use_raw = filegrp.get("USE")
            try:
                use = _UseType(use_raw)
            except ValueError:
                continue  # Ignore unknown USE types

            for file_elem in filegrp.xpath("./mets:file", namespaces=ns):
                file_id = file_elem.get("ID")
                mimetype = file_elem.get("MIMETYPE")
                flocat_elem = file_elem.find("mets:FLocat", namespaces=ns)
                href = (
                    flocat_elem.get("{http://www.w3.org/1999/xlink}href")
                    if flocat_elem is not None
                    else None
                )
                if href is None:
                    continue

                file_info_by_id[file_id] = _FileInfo(
                    file_id=file_id, mimetype=mimetype, path=href, use=use
                )

        USE_TO_ATTR = {
            _UseType.IMAGE: "image",
            _UseType.OCR: "ocr",
            _UseType.COORD_OCR: "coordOCR",
        }

        for div in self.root_mets.xpath('.//mets:div[@TYPE="page"]', namespaces=ns):
            order_str = div.get("ORDER")
            if not order_str:
                continue
            try:
                page_no = int(order_str) - 1  # make 0-index pages
            except ValueError:
                continue

            page_files = _PageFiles()

            for fptr in div.xpath("./mets:fptr", namespaces=ns):
                file_id = fptr.get("FILEID")
                file_info = file_info_by_id.get(file_id)

                if file_info:
                    attr = USE_TO_ATTR.get(file_info.use)
                    if attr:
                        setattr(page_files, attr, file_info)

            self.page_map[page_no] = page_files

    def _validate_mets_xml(self, xml_string) -> Optional[etree._Element]:
        root: etree._Element = etree.fromstring(xml_string)
        if (
            root.tag == "{http://www.loc.gov/METS/}mets"
            and root.get("PROFILE") == "gbs"
        ):
            return root

        _log.warning(f"The root element is not <mets:mets> with PROFILE='gbs': {root}")
        return None

    def _parse_page(self, page_no: int) -> Tuple[SegmentedPdfPage, PILImage]:
        # TODO: use better fallbacks...
        image_info = self.page_map[page_no].image
        assert image_info is not None
        ocr_info = self.page_map[page_no].coordOCR
        assert ocr_info is not None

        image_file = self._tar.extractfile(image_info.path)
        assert image_file is not None
        buf = BytesIO(image_file.read())
        im: PILImage = Image.open(buf)
        ocr_file = self._tar.extractfile(ocr_info.path)
        assert ocr_file is not None
        ocr_content = ocr_file.read()
        parser = etree.HTMLParser()
        ocr_root: etree._Element = etree.fromstring(ocr_content, parser=parser)

        line_cells: List[TextCell] = []
        word_cells: List[TextCell] = []

        page_div = ocr_root.xpath("//div[@class='ocr_page']")

        size = Size(width=im.size[0], height=im.size[1])
        if page_div:
            title = page_div[0].attrib.get("title", "")
            rect = _extract_rect(title)
            if rect:
                size = Size(width=rect.width, height=rect.height)
        else:
            _log.error(f"Could not find ocr_page for page {page_no}")

        im = im.resize(size=(round(size.width), round(size.height)))
        im = im.convert("RGB")

        # Extract all ocrx_word spans
        for ix, word in enumerate(ocr_root.xpath("//span[@class='ocrx_word']")):
            text = "".join(word.itertext()).strip()
            title = word.attrib.get("title", "")
            rect = _extract_rect(title)
            conf = _extract_confidence(title)
            if rect:
                word_cells.append(
                    TextCell(
                        index=ix,
                        text=text,
                        orig=text,
                        rect=rect,
                        from_ocr=True,
                        confidence=conf,
                    )
                )

        # Extract all ocr_line spans
        # line: etree._Element
        for ix, line in enumerate(ocr_root.xpath("//span[@class='ocr_line']")):
            text = "".join(line.itertext()).strip()
            title = line.attrib.get("title", "")
            rect = _extract_rect(title)
            conf = _extract_confidence(title)
            if rect:
                line_cells.append(
                    TextCell(
                        index=ix,
                        text=text,
                        orig=text,
                        rect=rect,
                        from_ocr=True,
                        confidence=conf,
                    )
                )

        page = SegmentedPdfPage(
            dimension=_get_pdf_page_geometry(size),
            textline_cells=line_cells,
            char_cells=[],
            word_cells=word_cells,
            has_textlines=True,
            has_words=True,
            has_chars=False,
        )
        return page, im

    def page_count(self) -> int:
        """Returns the total number of pages in the document."""
        return len(self.page_map)

    def load_page(self, page_no: int) -> MetsGbsPageBackend:
        """Loads a single page and returns a `MetsGbsPageBackend` for it.

        Args:
            page_no: The page number to load (0-indexed).

        Returns:
            A `MetsGbsPageBackend` instance for the specified page.
        """
        # TODO: is this thread-safe?
        page, im = self._parse_page(page_no)
        return MetsGbsPageBackend(parsed_page=page, page_im=im)

    def is_valid(self) -> bool:
        """Checks if the document is valid."""
        return self.root_mets is not None and self.page_count() > 0

    @classmethod
    def supported_formats(cls) -> Set[InputFormat]:
        """Returns the set of supported formats, which is just METS_GBS."""
        return {InputFormat.METS_GBS}

    @classmethod
    def supports_pagination(cls) -> bool:
        """METS GBS is a paginated format."""
        return True

    def unload(self) -> None:
        """Closes the underlying `tar.gz` archive."""
        super().unload()
        self._tar.close()
