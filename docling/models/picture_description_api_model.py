from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Type, Union

from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import (
    PictureDescriptionApiOptions,
    PictureDescriptionBaseOptions,
)
from docling.exceptions import OperationNotAllowed
from docling.models.picture_description_base_model import PictureDescriptionBaseModel
from docling.utils.api_image_request import api_image_request


class PictureDescriptionApiModel(PictureDescriptionBaseModel):
    """A model that uses an external API to generate descriptions for pictures.

    This class implements the `PictureDescriptionBaseModel` interface to generate
    textual descriptions for images by sending them to a remote API endpoint.

    Attributes:
        concurrency: The number of concurrent requests to make to the API.
    """

    # elements_batch_size = 4

    @classmethod
    def get_options_type(cls) -> Type[PictureDescriptionBaseOptions]:
        """Returns the options type for this model, which is `PictureDescriptionApiOptions`."""
        return PictureDescriptionApiOptions

    def __init__(
        self,
        enabled: bool,
        enable_remote_services: bool,
        artifacts_path: Optional[Union[Path, str]],
        options: PictureDescriptionApiOptions,
        accelerator_options: AcceleratorOptions,
    ):
        """Initializes the PictureDescriptionApiModel.

        Args:
            enabled: A boolean flag to enable or disable the model.
            enable_remote_services: A boolean flag that must be `True` to allow
                the model to make remote API calls.
            artifacts_path: An optional path to a directory for saving artifacts.
            options: The configuration options for the API model.
            accelerator_options: The hardware acceleration options.

        Raises:
            OperationNotAllowed: If the model is enabled but remote services
                are not.
        """
        super().__init__(
            enabled=enabled,
            enable_remote_services=enable_remote_services,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: PictureDescriptionApiOptions
        self.concurrency = self.options.concurrency

        if self.enabled:
            if not enable_remote_services:
                raise OperationNotAllowed(
                    "Connections to remote services is only allowed when set explicitly. "
                    "pipeline_options.enable_remote_services=True."
                )

    def _annotate_images(self, images: Iterable[Image.Image]) -> Iterable[str]:
        # Note: technically we could make a batch request here,
        # but not all APIs will allow for it. For example, vllm won't allow more than 1.
        def _api_request(image):
            return api_image_request(
                image=image,
                prompt=self.options.prompt,
                url=self.options.url,
                timeout=self.options.timeout,
                headers=self.options.headers,
                **self.options.params,
            )

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            yield from executor.map(_api_request, images)
