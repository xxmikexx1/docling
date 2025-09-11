import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Generic, Optional, Protocol, Type, Union

import numpy as np
from docling_core.types.doc import (
    BoundingBox,
    DocItem,
    DoclingDocument,
    NodeItem,
    PictureItem,
)
from PIL.Image import Image
from typing_extensions import TypeVar

from docling.datamodel.base_models import (
    ItemAndImageEnrichmentElement,
    Page,
    VlmPrediction,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import BaseOptions
from docling.datamodel.pipeline_options_vlm_model import (
    InlineVlmOptions,
    TransformersPromptStyle,
)
from docling.datamodel.settings import settings


class BaseModelWithOptions(Protocol):
    """A protocol for models that are initialized with an options object.

    This protocol defines the common interface for models that require a
    configuration object (derived from `BaseOptions`) for their initialization.
    """

    @classmethod
    def get_options_type(cls) -> Type[BaseOptions]:
        """Returns the type of the options class required by this model."""
        ...

    def __init__(self, *, options: BaseOptions, **kwargs):
        """Initializes the model with the given options."""
        ...


class BasePageModel(ABC):
    """An abstract base class for models that process pages of a document.

    This class defines the interface for models that operate on a batch of
    pages from a `ConversionResult`.
    """

    @abstractmethod
    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        """Processes a batch of pages.

        Args:
            conv_res: The `ConversionResult` object for the current document.
            page_batch: An iterable of `Page` objects to be processed.

        Returns:
            An iterable of the processed `Page` objects.
        """
        pass


class BaseVlmModel(ABC):
    """An abstract base class for Vision Language Models (VLMs).

    This class defines the core interface for VLMs, which is the ability to
    process a batch of images with a given prompt.
    """

    @abstractmethod
    def process_images(
        self,
        image_batch: Iterable[Union[Image, np.ndarray]],
        prompt: Union[str, list[str]],
    ) -> Iterable[VlmPrediction]:
        """Processes a batch of images with a given prompt or list of prompts.

        Args:
            image_batch: An iterable of PIL Images or numpy arrays.
            prompt: Either a single prompt string to be used for all images, or a
                list of prompts (one for each image).

        Returns:
            An iterable of `VlmPrediction` objects, one for each processed image.

        Raises:
            ValueError: If a list of prompts is provided and its length does not
                match the number of images.
        """


class BaseVlmPageModel(BasePageModel, BaseVlmModel):
    """An abstract base class for VLM-based models that process document pages.

    This class combines the interfaces of `BasePageModel` and `BaseVlmModel`,
    providing a foundation for models that extract images from pages, process
    them with a VLM, and attach the results back to the pages.
    """

    # Type annotations for attributes that subclasses must initialize
    vlm_options: InlineVlmOptions
    processor: Any

    @abstractmethod
    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        """Extract images from pages, process them, and attach results back."""

    def formulate_prompt(self, user_prompt: str) -> str:
        """Constructs a model-specific prompt from a user-provided prompt.

        This method takes a user-provided prompt and wraps it in the appropriate
        template for the specific VLM being used, such as applying a chat
        template or adding special tokens.

        Args:
            user_prompt: The user-facing prompt string.

        Returns:
            The fully formatted prompt ready to be sent to the model.

        Raises:
            RuntimeError: If the prompt style specified in the options is unknown.
        """
        _log = logging.getLogger(__name__)

        if self.vlm_options.transformers_prompt_style == TransformersPromptStyle.RAW:
            return user_prompt

        elif self.vlm_options.repo_id == "microsoft/Phi-4-multimodal-instruct":
            _log.debug("Using specialized prompt for Phi-4")
            # Note: This might need adjustment for VLLM vs transformers
            user_prompt_prefix = "<|user|>"
            assistant_prompt = "<|assistant|>"
            prompt_suffix = "<|end|>"

            prompt = f"{user_prompt_prefix}<|image_1|>{user_prompt}{prompt_suffix}{assistant_prompt}"
            _log.debug(f"prompt for {self.vlm_options.repo_id}: {prompt}")

            return prompt

        elif self.vlm_options.transformers_prompt_style == TransformersPromptStyle.CHAT:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "This is a page from a document.",
                        },
                        {"type": "image"},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            return prompt

        raise RuntimeError(
            f"Unknown prompt style `{self.vlm_options.transformers_prompt_style}`. Valid values are {', '.join(s.value for s in TransformersPromptStyle)}."
        )


EnrichElementT = TypeVar("EnrichElementT", default=NodeItem)


class GenericEnrichmentModel(ABC, Generic[EnrichElementT]):
    """An abstract base class for models that enrich document elements.

    This generic class defines the interface for enrichment models, which take
    document elements, process them, and return them with additional information
    or modifications. It supports batching of elements for efficiency.

    TypeVar:
        EnrichElementT: The type of the element to be processed by the model.
    """

    elements_batch_size: int = settings.perf.elements_batch_size

    @abstractmethod
    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        """Checks if a given document element can be processed by this model."""
        pass

    @abstractmethod
    def prepare_element(
        self, conv_res: ConversionResult, element: NodeItem
    ) -> Optional[EnrichElementT]:
        """Prepares an element for processing.

        This may involve extracting data, transforming the element, or simply
        checking if it's processable.

        Returns:
            The prepared element, or `None` if it cannot be prepared.
        """
        pass

    @abstractmethod
    def __call__(
        self, doc: DoclingDocument, element_batch: Iterable[EnrichElementT]
    ) -> Iterable[NodeItem]:
        """Processes a batch of prepared elements.

        Args:
            doc: The `DoclingDocument` being processed.
            element_batch: An iterable of prepared elements to be enriched.

        Returns:
            An iterable of the enriched `NodeItem`s.
        """
        pass


class BaseEnrichmentModel(GenericEnrichmentModel[NodeItem]):
    """A base implementation of `GenericEnrichmentModel` for `NodeItem`s.

    This class provides a default implementation of the `prepare_element` method
    that simply checks if the element is processable and returns it if so.
    """

    def prepare_element(
        self, conv_res: ConversionResult, element: NodeItem
    ) -> Optional[NodeItem]:
        """Prepares a `NodeItem` for processing.

        Args:
            conv_res: The `ConversionResult` for the current document.
            element: The `NodeItem` to prepare.

        Returns:
            The element if it is processable, otherwise `None`.
        """
        if self.is_processable(doc=conv_res.document, element=element):
            return element
        return None


class BaseItemAndImageEnrichmentModel(
    GenericEnrichmentModel[ItemAndImageEnrichmentElement]
):
    """A base class for enrichment models that process an item and its corresponding image.

    This class provides a default implementation for preparing an element by
    extracting its image from the page, potentially with an expanded bounding
    box.

    Attributes:
        images_scale: The scaling factor to apply to the extracted image.
        expansion_factor: A factor to expand the element's bounding box by
            before cropping the image.
    """

    images_scale: float
    expansion_factor: float = 0.0

    def prepare_element(
        self, conv_res: ConversionResult, element: NodeItem
    ) -> Optional[ItemAndImageEnrichmentElement]:
        """Prepares an element by extracting its image.

        This method crops the image of the element from the page image,
        optionally expanding the bounding box. If a page image is not
        available, it tries to get an embedded image from the element itself.

        Args:
            conv_res: The `ConversionResult` for the current document.
            element: The `NodeItem` to prepare.

        Returns:
            An `ItemAndImageEnrichmentElement` containing the item and its
            image, or `None` if the image cannot be obtained.
        """
        if not self.is_processable(doc=conv_res.document, element=element):
            return None

        assert isinstance(element, DocItem)

        # Allow the case of documents without page images but embedded images (e.g. Word and HTML docs)
        if len(element.prov) == 0 and isinstance(element, PictureItem):
            embedded_im = element.get_image(conv_res.document)
            if embedded_im is not None:
                return ItemAndImageEnrichmentElement(item=element, image=embedded_im)
            else:
                return None

        # Crop the image form the page
        element_prov = element.prov[0]
        bbox = element_prov.bbox
        width = bbox.r - bbox.l
        height = bbox.t - bbox.b

        # TODO: move to a utility in the BoundingBox class
        expanded_bbox = BoundingBox(
            l=bbox.l - width * self.expansion_factor,
            t=bbox.t + height * self.expansion_factor,
            r=bbox.r + width * self.expansion_factor,
            b=bbox.b - height * self.expansion_factor,
            coord_origin=bbox.coord_origin,
        )

        page_ix = element_prov.page_no - conv_res.pages[0].page_no - 1
        cropped_image = conv_res.pages[page_ix].get_image(
            scale=self.images_scale, cropbox=expanded_bbox
        )

        # Allow for images being embedded without the page backend or page images
        if cropped_image is None and isinstance(element, PictureItem):
            embedded_im = element.get_image(conv_res.document)
            if embedded_im is not None:
                return ItemAndImageEnrichmentElement(item=element, image=embedded_im)
            else:
                return None

        # Return the proper cropped image
        return ItemAndImageEnrichmentElement(item=element, image=cropped_image)
