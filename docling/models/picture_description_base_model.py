from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import List, Optional, Type, Union

from docling_core.types.doc import (
    DoclingDocument,
    NodeItem,
    PictureItem,
)
from docling_core.types.doc.document import (  # TODO: move import to docling_core.types.doc
    PictureDescriptionData,
)
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import (
    PictureDescriptionBaseOptions,
)
from docling.models.base_model import (
    BaseItemAndImageEnrichmentModel,
    BaseModelWithOptions,
    ItemAndImageEnrichmentElement,
)


class PictureDescriptionBaseModel(
    BaseItemAndImageEnrichmentModel, BaseModelWithOptions
):
    """An abstract base class for picture description models.

    This class provides a common framework for models that generate textual
    descriptions for images. It handles the processing of `PictureItem` elements,
    filtering them based on size, and attaching the generated descriptions as
    annotations.

    Attributes:
        enabled: A boolean indicating if the model is enabled.
        options: A `PictureDescriptionBaseOptions` object for configuration.
        provenance: A string identifying the source of the description.
    """

    images_scale: float = 2.0

    def __init__(
        self,
        *,
        enabled: bool,
        enable_remote_services: bool,
        artifacts_path: Optional[Union[Path, str]],
        options: PictureDescriptionBaseOptions,
        accelerator_options: AcceleratorOptions,
    ):
        """Initializes the PictureDescriptionBaseModel.

        Args:
            enabled: A boolean flag to enable or disable the model.
            enable_remote_services: A boolean flag that must be `True` to allow
                the model to make remote API calls.
            artifacts_path: An optional path to a directory for saving artifacts.
            options: The configuration options for the model.
            accelerator_options: The hardware acceleration options.
        """
        self.enabled = enabled
        self.options = options
        self.provenance = "not-implemented"

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        """Determines if a given element can be processed by this model.

        This model can only process `PictureItem` elements.

        Args:
            doc: The `DoclingDocument` being processed.
            element: The `NodeItem` to check.

        Returns:
            `True` if the element is a `PictureItem`, `False` otherwise.
        """
        return self.enabled and isinstance(element, PictureItem)

    def _annotate_images(self, images: Iterable[Image.Image]) -> Iterable[str]:
        raise NotImplementedError

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        """Processes a batch of picture elements and adds descriptions as annotations.

        Args:
            doc: The `DoclingDocument` being processed.
            element_batch: An iterable of picture elements to be described.

        Returns:
            An iterable of the enriched `PictureItem`s with descriptions added.
        """
        if not self.enabled:
            for element in element_batch:
                yield element.item
            return

        images: List[Image.Image] = []
        elements: List[PictureItem] = []
        for el in element_batch:
            assert isinstance(el.item, PictureItem)
            describe_image = True
            # Don't describe the image if it's smaller than the threshold
            if len(el.item.prov) > 0:
                prov = el.item.prov[0]  # PictureItems have at most a single provenance
                page = doc.pages.get(prov.page_no)
                if page is not None:
                    page_area = page.size.width * page.size.height
                    if page_area > 0:
                        area_fraction = prov.bbox.area() / page_area
                        if area_fraction < self.options.picture_area_threshold:
                            describe_image = False
            if describe_image:
                elements.append(el.item)
                images.append(el.image)

        outputs = self._annotate_images(images)

        for item, output in zip(elements, outputs):
            item.annotations.append(
                PictureDescriptionData(text=output, provenance=self.provenance)
            )
            yield item

    @classmethod
    @abstractmethod
    def get_options_type(cls) -> Type[PictureDescriptionBaseOptions]:
        """Returns the options type for this model."""
        pass
