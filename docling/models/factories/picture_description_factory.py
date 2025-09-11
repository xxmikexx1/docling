import logging

from docling.models.factories.base_factory import BaseFactory
from docling.models.picture_description_base_model import PictureDescriptionBaseModel

logger = logging.getLogger(__name__)


class PictureDescriptionFactory(BaseFactory[PictureDescriptionBaseModel]):
    """A factory for creating instances of picture description models.

    This class extends `BaseFactory` to specifically handle the creation of
    picture description models. It looks for the "picture_description" attribute
    in plugins to discover and register different model implementations.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the PictureDescriptionFactory."""
        super().__init__("picture_description", *args, **kwargs)
