import copy
import logging
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from docling_core.types.doc import DocItemLabel
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import BoundingBox, Cluster, LayoutPrediction, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_V2, LayoutModelConfig
from docling.datamodel.pipeline_options import LayoutOptions
from docling.datamodel.settings import settings
from docling.models.base_model import BasePageModel
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device
from docling.utils.layout_postprocessor import LayoutPostprocessor
from docling.utils.profiling import TimeRecorder
from docling.utils.visualization import draw_clusters

_log = logging.getLogger(__name__)


class LayoutModel(BasePageModel):
    """A model for performing layout analysis on document pages.

    This model uses a layout prediction model to identify and classify different
    regions on a page, such as text blocks, tables, figures, and formulas. It
    then uses a `LayoutPostprocessor` to refine these predictions.

    Attributes:
        options: A `LayoutOptions` object for configuration.
        layout_predictor: An instance of the underlying layout prediction model.
    """

    TEXT_ELEM_LABELS = [
        DocItemLabel.TEXT,
        DocItemLabel.FOOTNOTE,
        DocItemLabel.CAPTION,
        DocItemLabel.CHECKBOX_UNSELECTED,
        DocItemLabel.CHECKBOX_SELECTED,
        DocItemLabel.SECTION_HEADER,
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.PAGE_FOOTER,
        DocItemLabel.CODE,
        DocItemLabel.LIST_ITEM,
        DocItemLabel.FORMULA,
    ]
    PAGE_HEADER_LABELS = [DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER]

    TABLE_LABELS = [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]
    FIGURE_LABEL = DocItemLabel.PICTURE
    FORMULA_LABEL = DocItemLabel.FORMULA
    CONTAINER_LABELS = [DocItemLabel.FORM, DocItemLabel.KEY_VALUE_REGION]

    def __init__(
        self,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        options: LayoutOptions,
    ):
        """Initializes the LayoutModel.

        Args:
            artifacts_path: An optional path to the directory containing the
                model artifacts. If not provided, the model will be downloaded.
            accelerator_options: The hardware acceleration options.
            options: The configuration options for the layout model.
        """
        from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor

        self.options = options

        device = decide_device(accelerator_options.device)
        layout_model_config = options.model_spec
        model_repo_folder = layout_model_config.model_repo_folder
        model_path = layout_model_config.model_path

        if artifacts_path is None:
            artifacts_path = (
                self.download_models(layout_model_config=layout_model_config)
                / model_path
            )
        else:
            if (artifacts_path / model_repo_folder).exists():
                artifacts_path = artifacts_path / model_repo_folder / model_path
            elif (artifacts_path / model_path).exists():
                warnings.warn(
                    "The usage of artifacts_path containing directly "
                    f"{model_path} is deprecated. Please point "
                    "the artifacts_path to the parent containing "
                    f"the {model_repo_folder} folder.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                artifacts_path = artifacts_path / model_path

        self.layout_predictor = LayoutPredictor(
            artifact_path=str(artifacts_path),
            device=device,
            num_threads=accelerator_options.num_threads,
        )

    @staticmethod
    def download_models(
        local_dir: Optional[Path] = None,
        force: bool = False,
        progress: bool = False,
        layout_model_config: LayoutModelConfig = LayoutOptions().model_spec,  # use default
    ) -> Path:
        """Downloads the layout model from Hugging Face.

        Args:
            local_dir: An optional local directory to save the model to.
            force: If `True`, forces the re-download of the model.
            progress: If `True`, displays a progress bar.
            layout_model_config: The configuration for the layout model to download.

        Returns:
            The path to the local directory where the model is saved.
        """
        return download_hf_model(
            repo_id=layout_model_config.repo_id,
            revision=layout_model_config.revision,
            local_dir=local_dir,
            force=force,
            progress=progress,
        )

    def draw_clusters_and_cells_side_by_side(
        self, conv_res, page, clusters, mode_prefix: str, show: bool = False
    ):
        """Generates a debug image visualizing the layout analysis results.

        This method creates a side-by-side comparison image showing the page
        with different sets of clusters drawn on it. This is useful for
        debugging the layout analysis and post-processing steps.

        Args:
            conv_res: The `ConversionResult` for the current document.
            page: The `Page` object that was processed.
            clusters: The list of `Cluster` objects to visualize.
            mode_prefix: A prefix for the output filename (e.g., "raw", "postprocessed").
            show: If `True`, displays the image instead of saving it to a file.
        """
        scale_x = page.image.width / page.size.width
        scale_y = page.image.height / page.size.height

        # Filter clusters for left and right images
        exclude_labels = {
            DocItemLabel.FORM,
            DocItemLabel.KEY_VALUE_REGION,
            DocItemLabel.PICTURE,
        }
        left_clusters = [c for c in clusters if c.label not in exclude_labels]
        right_clusters = [c for c in clusters if c.label in exclude_labels]
        # Create a deep copy of the original image for both sides
        left_image = page.image.copy()
        right_image = page.image.copy()

        # Draw clusters on both images
        draw_clusters(left_image, left_clusters, scale_x, scale_y)
        draw_clusters(right_image, right_clusters, scale_x, scale_y)
        # Combine the images side by side
        combined_width = left_image.width * 2
        combined_height = left_image.height
        combined_image = Image.new("RGB", (combined_width, combined_height))
        combined_image.paste(left_image, (0, 0))
        combined_image.paste(right_image, (left_image.width, 0))
        if show:
            combined_image.show()
        else:
            out_path: Path = (
                Path(settings.debug.debug_output_path)
                / f"debug_{conv_res.input.file.stem}"
            )
            out_path.mkdir(parents=True, exist_ok=True)
            out_file = out_path / f"{mode_prefix}_layout_page_{page.page_no:05}.png"
            combined_image.save(str(out_file), format="png")

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        """Processes a batch of pages, performing layout analysis on each.

        Args:
            conv_res: The `ConversionResult` for the current document.
            page_batch: An iterable of `Page` objects to be processed.

        Yields:
            The processed `Page` objects with layout predictions attached.
        """
        # Convert to list to allow multiple iterations
        pages = list(page_batch)

        # Separate valid and invalid pages
        valid_pages = []
        valid_page_images: List[Union[Image.Image, np.ndarray]] = []

        for page in pages:
            assert page._backend is not None
            if not page._backend.is_valid():
                continue

            assert page.size is not None
            page_image = page.get_image(scale=1.0)
            assert page_image is not None

            valid_pages.append(page)
            valid_page_images.append(page_image)

        # Process all valid pages with batch prediction
        batch_predictions = []
        if valid_page_images:
            with TimeRecorder(conv_res, "layout"):
                batch_predictions = self.layout_predictor.predict_batch(  # type: ignore[attr-defined]
                    valid_page_images
                )

        # Process each page with its predictions
        valid_page_idx = 0
        for page in pages:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
                continue

            page_predictions = batch_predictions[valid_page_idx]
            valid_page_idx += 1

            clusters = []
            for ix, pred_item in enumerate(page_predictions):
                label = DocItemLabel(
                    pred_item["label"].lower().replace(" ", "_").replace("-", "_")
                )  # Temporary, until docling-ibm-model uses docling-core types
                cluster = Cluster(
                    id=ix,
                    label=label,
                    confidence=pred_item["confidence"],
                    bbox=BoundingBox.model_validate(pred_item),
                    cells=[],
                )
                clusters.append(cluster)

            if settings.debug.visualize_raw_layout:
                self.draw_clusters_and_cells_side_by_side(
                    conv_res, page, clusters, mode_prefix="raw"
                )

            # Apply postprocessing
            processed_clusters, processed_cells = LayoutPostprocessor(
                page, clusters, self.options
            ).postprocess()
            # Note: LayoutPostprocessor updates page.cells and page.parsed_page internally

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "Mean of empty slice|invalid value encountered in scalar divide",
                    RuntimeWarning,
                    "numpy",
                )

                conv_res.confidence.pages[page.page_no].layout_score = float(
                    np.mean([c.confidence for c in processed_clusters])
                )

                conv_res.confidence.pages[page.page_no].ocr_score = float(
                    np.mean([c.confidence for c in processed_cells if c.from_ocr])
                )

            page.predictions.layout = LayoutPrediction(clusters=processed_clusters)

            if settings.debug.visualize_layout:
                self.draw_clusters_and_cells_side_by_side(
                    conv_res, page, processed_clusters, mode_prefix="postprocessed"
                )

            yield page
