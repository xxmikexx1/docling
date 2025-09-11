import logging

from docling.models.base_ocr_model import BaseOcrModel
from docling.models.factories.base_factory import BaseFactory

logger = logging.getLogger(__name__)


class OcrFactory(BaseFactory[BaseOcrModel]):
    """A factory for creating instances of OCR models.

    This class extends `BaseFactory` to specifically handle the creation of
    OCR models. It looks for the "ocr_engines" attribute in plugins to discover
    and register different OCR model implementations.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the OcrFactory."""
        super().__init__("ocr_engines", *args, **kwargs)
