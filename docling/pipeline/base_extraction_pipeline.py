import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from docling.datamodel.base_models import ConversionStatus, ErrorItem
from docling.datamodel.document import InputDocument
from docling.datamodel.extraction import ExtractionResult, ExtractionTemplateType
from docling.datamodel.pipeline_options import BaseOptions, PipelineOptions
from docling.datamodel.settings import settings

_log = logging.getLogger(__name__)


class BaseExtractionPipeline(ABC):
    """Abstract base class for all extraction pipelines.

    This class provides the basic structure for pipelines that extract structured
    data from documents. It handles initialization, artifact path management,
    and the main execution flow.

    Attributes:
        pipeline_options: The configuration options for the pipeline.
        artifacts_path: The path to the directory containing model artifacts.
    """

    def __init__(self, pipeline_options: PipelineOptions):
        """Initializes the BaseExtractionPipeline.

        Args:
            pipeline_options: The configuration options for the pipeline.

        Raises:
            RuntimeError: If the specified artifacts_path is not a valid directory.
        """
        self.pipeline_options = pipeline_options

        self.artifacts_path: Optional[Path] = None
        if pipeline_options.artifacts_path is not None:
            self.artifacts_path = Path(pipeline_options.artifacts_path).expanduser()
        elif settings.artifacts_path is not None:
            self.artifacts_path = Path(settings.artifacts_path).expanduser()

        if self.artifacts_path is not None and not self.artifacts_path.is_dir():
            raise RuntimeError(
                f"The value of {self.artifacts_path=} is not valid. "
                "When defined, it must point to a folder containing all models required by the pipeline."
            )

    def execute(
        self,
        in_doc: InputDocument,
        raises_on_error: bool,
        template: Optional[ExtractionTemplateType] = None,
    ) -> ExtractionResult:
        """Executes the extraction pipeline.

        This method orchestrates the data extraction process, including error handling.

        Args:
            in_doc: The input document to process.
            raises_on_error: If True, exceptions will be raised. Otherwise, they
                will be caught and recorded in the result.
            template: The extraction template to use.

        Returns:
            An ExtractionResult object containing the extracted data and status.
        """
        ext_res = ExtractionResult(input=in_doc)

        try:
            ext_res = self._extract_data(ext_res, template)
            ext_res.status = self._determine_status(ext_res)
        except Exception as e:
            ext_res.status = ConversionStatus.FAILURE
            error_item = ErrorItem(
                component_type="extraction_pipeline",
                module_name=self.__class__.__name__,
                error_message=str(e),
            )
            ext_res.errors.append(error_item)
            if raises_on_error:
                raise e

        return ext_res

    @abstractmethod
    def _extract_data(
        self,
        ext_res: ExtractionResult,
        template: Optional[ExtractionTemplateType] = None,
    ) -> ExtractionResult:
        """Abstract method for the main data extraction logic.

        Subclasses must implement this method to populate the `ext_res` object
        with extracted data (e.g., pages, errors).

        Args:
            ext_res: The ExtractionResult object to populate.
            template: The extraction template to use.

        Returns:
            The populated ExtractionResult object.
        """
        raise NotImplementedError

    @abstractmethod
    def _determine_status(self, ext_res: ExtractionResult) -> ConversionStatus:
        """Determines the final status of the extraction.

        Subclasses must implement this method to decide the final status
        (e.g., SUCCESS, FAILURE) based on the content of `ext_res`.

        Args:
            ext_res: The ExtractionResult object.

        Returns:
            The final ConversionStatus.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_default_options(cls) -> PipelineOptions:
        """Gets the default pipeline options for this extraction pipeline.

        Returns:
            A PipelineOptions object with default values.
        """
        pass
