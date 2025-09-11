import logging
from io import BytesIO
from pathlib import Path
from typing import Set, Union

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


class NoOpBackend(AbstractDocumentBackend):
    """A no-op backend that performs no conversion but validates input existence.

    This backend is used as a placeholder for file formats, such as audio, where
    the actual processing is handled by a specialized pipeline (e.g., ASR) rather
    than a traditional document conversion backend. Its primary role is to
    validate that the input file or stream exists and is not empty.
    """

    def __init__(self, in_doc: "InputDocument", path_or_stream: Union[BytesIO, Path]):
        """Initializes the NoOpBackend.

        Args:
            in_doc: The `InputDocument` object.
            path_or_stream: The path or stream of the input file.
        """
        super().__init__(in_doc, path_or_stream)

        _log.debug(f"NoOpBackend initialized for: {path_or_stream}")

        # Validate input
        try:
            if isinstance(self.path_or_stream, BytesIO):
                # Check if stream has content
                self.valid = len(self.path_or_stream.getvalue()) > 0
                _log.debug(
                    f"BytesIO stream length: {len(self.path_or_stream.getvalue())}"
                )
            elif isinstance(self.path_or_stream, Path):
                # Check if file exists
                self.valid = self.path_or_stream.exists()
                _log.debug(f"File exists: {self.valid}")
            else:
                self.valid = False
        except Exception as e:
            _log.error(f"NoOpBackend validation failed: {e}")
            self.valid = False

    def is_valid(self) -> bool:
        """Checks if the input file or stream is valid (i.e., exists and is not empty)."""
        return self.valid

    @classmethod
    def supports_pagination(cls) -> bool:
        """This backend does not support pagination."""
        return False

    @classmethod
    def supported_formats(cls) -> Set[InputFormat]:
        """This backend notionally supports all input formats."""
        return set(InputFormat)
