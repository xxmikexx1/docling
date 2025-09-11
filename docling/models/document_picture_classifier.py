from collections.abc import Iterable
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
from docling_core.types.doc import (
    DoclingDocument,
    NodeItem,
    PictureClassificationClass,
    PictureClassificationData,
    PictureItem,
)
from PIL import Image
from pydantic import BaseModel

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device


class DocumentPictureClassifierOptions(BaseModel):
    """Configuration options for the `DocumentPictureClassifier`.

    Attributes:
        kind: The type of the model, fixed to "document_picture_classifier".
    """

    kind: Literal["document_picture_classifier"] = "document_picture_classifier"


class DocumentPictureClassifier(BaseItemAndImageEnrichmentModel):
    """A model for classifying pictures within a document.

    This model enriches `PictureItem` elements with predicted classifications
    (e.g., "photograph", "diagram", "table") based on a pre-trained model.

    Attributes:
        enabled: A boolean indicating if the model is enabled.
        options: A `DocumentPictureClassifierOptions` object for configuration.
        document_picture_classifier: An instance of the underlying predictor model.
    """

    _model_repo_folder = "ds4sd--DocumentFigureClassifier"
    images_scale = 2

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: DocumentPictureClassifierOptions,
        accelerator_options: AcceleratorOptions,
    ):
        """Initializes the DocumentPictureClassifier.

        Args:
            enabled: A boolean flag to enable or disable the model.
            artifacts_path: An optional path to the directory containing the
                model artifacts. If not provided, the model will be downloaded.
            options: The configuration options for the classifier.
            accelerator_options: The hardware acceleration options.
        """
        self.enabled = enabled
        self.options = options

        if self.enabled:
            device = decide_device(accelerator_options.device)
            from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor import (
                DocumentFigureClassifierPredictor,
            )

            if artifacts_path is None:
                artifacts_path = self.download_models()
            else:
                artifacts_path = artifacts_path / self._model_repo_folder

            self.document_picture_classifier = DocumentFigureClassifierPredictor(
                artifacts_path=str(artifacts_path),
                device=device,
                num_threads=accelerator_options.num_threads,
            )

    @staticmethod
    def download_models(
        local_dir: Optional[Path] = None, force: bool = False, progress: bool = False
    ) -> Path:
        """Downloads the DocumentFigureClassifier model from Hugging Face.

        Args:
            local_dir: An optional local directory to save the model to.
            force: If `True`, forces the re-download of the model.
            progress: If `True`, displays a progress bar.

        Returns:
            The path to the local directory where the model is saved.
        """
        return download_hf_model(
            repo_id="ds4sd/DocumentFigureClassifier",
            revision="v1.0.1",
            local_dir=local_dir,
            force=force,
            progress=progress,
        )

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

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        """Processes a batch of picture elements and adds classification predictions.

        Args:
            doc: The `DoclingDocument` being processed.
            element_batch: An iterable of picture elements to be classified.

        Returns:
            An iterable of the enriched `PictureItem`s with classification
            annotations added.
        """
        if not self.enabled:
            for element in element_batch:
                yield element.item
            return

        images: List[Union[Image.Image, np.ndarray]] = []
        elements: List[PictureItem] = []
        for el in element_batch:
            assert isinstance(el.item, PictureItem)
            elements.append(el.item)
            images.append(el.image)

        outputs = self.document_picture_classifier.predict(images)

        for item, output in zip(elements, outputs):
            item.annotations.append(
                PictureClassificationData(
                    provenance="DocumentPictureClassifier",
                    predicted_classes=[
                        PictureClassificationClass(
                            class_name=pred[0],
                            confidence=pred[1],
                        )
                        for pred in output
                    ],
                )
            )

            yield item
