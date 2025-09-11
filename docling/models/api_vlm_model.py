from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from docling.datamodel.base_models import Page, VlmPrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions
from docling.exceptions import OperationNotAllowed
from docling.models.base_model import BasePageModel
from docling.utils.api_image_request import api_image_request
from docling.utils.profiling import TimeRecorder


class ApiVlmModel(BasePageModel):
    """A model that uses an external API for Vision Language Model (VLM) processing.

    This class implements the `BasePageModel` interface to process pages by
    sending their images to a remote VLM service via an API call. It handles
    the construction of the API request and the parsing of the response.

    Attributes:
        enabled: A boolean indicating if the model is enabled.
        vlm_options: An `ApiVlmOptions` object containing the configuration for
            the API-based VLM.
        timeout: The request timeout in seconds.
        concurrency: The number of concurrent requests to make to the API.
        params: A dictionary of parameters to include in the API request.
    """

    def __init__(
        self,
        enabled: bool,
        enable_remote_services: bool,
        vlm_options: ApiVlmOptions,
    ):
        """Initializes the ApiVlmModel.

        Args:
            enabled: A boolean flag to enable or disable the model.
            enable_remote_services: A boolean flag that must be `True` to allow
                the model to make remote API calls.
            vlm_options: The configuration options for the API VLM.

        Raises:
            OperationNotAllowed: If the model is enabled but remote services
                are not.
        """
        self.enabled = enabled
        self.vlm_options = vlm_options
        if self.enabled:
            if not enable_remote_services:
                raise OperationNotAllowed(
                    "Connections to remote services is only allowed when set explicitly. "
                    "pipeline_options.enable_remote_services=True, or using the CLI "
                    "--enable-remote-services."
                )

            self.timeout = self.vlm_options.timeout
            self.concurrency = self.vlm_options.concurrency
            self.params = {
                **self.vlm_options.params,
                "temperature": self.vlm_options.temperature,
            }

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        """Processes a batch of pages by making API calls to a VLM service.

        This method uses a thread pool to concurrently process the pages in the
        batch. For each page, it renders an image, sends it to the configured
        API endpoint, and attaches the VLM's response to the page's predictions.

        Args:
            conv_res: The `ConversionResult` for the current document.
            page_batch: An iterable of `Page` objects to be processed.

        Yields:
            The processed `Page` objects with VLM predictions attached.
        """

        def _vlm_request(page):
            assert page._backend is not None
            if not page._backend.is_valid():
                return page
            else:
                with TimeRecorder(conv_res, "vlm"):
                    assert page.size is not None

                    hi_res_image = page.get_image(
                        scale=self.vlm_options.scale, max_size=self.vlm_options.max_size
                    )
                    assert hi_res_image is not None
                    if hi_res_image:
                        if hi_res_image.mode != "RGB":
                            hi_res_image = hi_res_image.convert("RGB")

                    prompt = self.vlm_options.build_prompt(page.parsed_page)
                    page_tags = api_image_request(
                        image=hi_res_image,
                        prompt=prompt,
                        url=self.vlm_options.url,
                        timeout=self.timeout,
                        headers=self.vlm_options.headers,
                        **self.params,
                    )

                    page_tags = self.vlm_options.decode_response(page_tags)
                    page.predictions.vlm_response = VlmPrediction(text=page_tags)

                return page

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            yield from executor.map(_vlm_request, page_batch)
