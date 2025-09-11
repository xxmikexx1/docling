from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Set, Union

from docling_core.types.doc import DoclingDocument

if TYPE_CHECKING:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import InputDocument


class AbstractDocumentBackend(ABC):
    """An abstract base class for all document processing backends.

    This class defines the common interface that all document backends must
    implement. A backend is responsible for parsing a specific document format
    and providing access to its content.

    Attributes:
        file: The path to the document file.
        path_or_stream: The source of the document, either a path or a stream.
        document_hash: The hash of the document's content.
        input_format: The `InputFormat` of the document.
    """

    @abstractmethod
    def __init__(self, in_doc: "InputDocument", path_or_stream: Union[BytesIO, Path]):
        """Initializes the document backend.

        Args:
            in_doc: The `InputDocument` object representing the source document.
            path_or_stream: The path or stream of the document content.
        """
        self.file = in_doc.file
        self.path_or_stream = path_or_stream
        self.document_hash = in_doc.document_hash
        self.input_format = in_doc.format

    @abstractmethod
    def is_valid(self) -> bool:
        """Checks if the document is valid and can be processed by this backend."""
        pass

    @classmethod
    @abstractmethod
    def supports_pagination(cls) -> bool:
        """Returns `True` if the backend supports paginated access to the document."""
        pass

    def unload(self):
        """Releases any resources held by the backend.

        This method should be called when the backend is no longer needed to
        ensure that file handles and other resources are properly closed.
        """
        if isinstance(self.path_or_stream, BytesIO):
            self.path_or_stream.close()

        self.path_or_stream = None

    @classmethod
    @abstractmethod
    def supported_formats(cls) -> Set["InputFormat"]:
        """Returns a set of `InputFormat` enums that this backend supports."""
        pass


class PaginatedDocumentBackend(AbstractDocumentBackend):
    """An abstract base class for backends that support paginated documents.

    This class extends `AbstractDocumentBackend` with an abstract method for
    retrieving the total number of pages in a document.
    """

    @abstractmethod
    def page_count(self) -> int:
        """Returns the total number of pages in the document."""
        pass


class DeclarativeDocumentBackend(AbstractDocumentBackend):
    """An abstract base class for backends that can directly convert to a `DoclingDocument`.

    This class is for backends that handle formats with a clear, declarative
    structure (like HTML, Markdown, or JATS XML). These backends can transform
    the source directly into a `DoclingDocument` without needing a complex
    recognition pipeline involving layout analysis or OCR.
    """

    @abstractmethod
    def convert(self) -> DoclingDocument:
        """Converts the source document into a `DoclingDocument`.

        Returns:
            A `DoclingDocument` object representing the content and structure
            of the source document.
        """
        pass
