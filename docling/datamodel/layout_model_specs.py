import logging
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from docling.datamodel.accelerator_options import AcceleratorDevice

_log = logging.getLogger(__name__)


class LayoutModelConfig(BaseModel):
    """Configuration for a layout analysis model.

    This class defines the necessary parameters to identify, download, and use a
    layout detection model from a repository, typically Hugging Face.

    Attributes:
        name: A unique name for the layout model.
        repo_id: The repository ID on Hugging Face (e.g., "ds4sd/docling-layout-heron").
        revision: The specific revision or branch of the model to use (e.g., "main").
        model_path: The local path to the model artifacts.
        supported_devices: A list of `AcceleratorDevice` enums indicating which
            hardware accelerators are compatible with this model.
        model_repo_folder: A property that generates a local folder name from the
            repo_id for caching.
    """

    name: str
    repo_id: str
    revision: str
    model_path: str
    supported_devices: list[AcceleratorDevice] = [
        AcceleratorDevice.CPU,
        AcceleratorDevice.CUDA,
        AcceleratorDevice.MPS,
    ]

    @property
    def model_repo_folder(self) -> str:
        return self.repo_id.replace("/", "--")


# HuggingFace Layout Models

# Default Docling Layout Model
DOCLING_LAYOUT_V2 = LayoutModelConfig(
    name="docling_layout_v2",
    repo_id="ds4sd/docling-layout-old",
    revision="main",
    model_path="",
)

DOCLING_LAYOUT_HERON = LayoutModelConfig(
    name="docling_layout_heron",
    repo_id="ds4sd/docling-layout-heron",
    revision="main",
    model_path="",
)

DOCLING_LAYOUT_HERON_101 = LayoutModelConfig(
    name="docling_layout_heron_101",
    repo_id="ds4sd/docling-layout-heron-101",
    revision="main",
    model_path="",
)

DOCLING_LAYOUT_EGRET_MEDIUM = LayoutModelConfig(
    name="docling_layout_egret_medium",
    repo_id="ds4sd/docling-layout-egret-medium",
    revision="main",
    model_path="",
)

DOCLING_LAYOUT_EGRET_LARGE = LayoutModelConfig(
    name="docling_layout_egret_large",
    repo_id="ds4sd/docling-layout-egret-large",
    revision="main",
    model_path="",
)

DOCLING_LAYOUT_EGRET_XLARGE = LayoutModelConfig(
    name="docling_layout_egret_xlarge",
    repo_id="ds4sd/docling-layout-egret-xlarge",
    revision="main",
    model_path="",
)

# Example for a hypothetical alternative model
# ALTERNATIVE_LAYOUT = LayoutModelConfig(
#     name="alternative_layout",
#     repo_id="someorg/alternative-layout",
#     revision="main",
#     model_path="model_artifacts/layout_alt",
# )


class LayoutModelType(str, Enum):
    """An enumeration of supported layout analysis models.

    This enum provides a standardized set of identifiers for the different layout
    detection models available in Docling. This allows users to easily select a
    specific model for their document processing pipeline.

    Attributes:
        DOCLING_LAYOUT_V2: The second version of the Docling layout model.
        DOCLING_LAYOUT_HERON: The Heron layout model.
        DOCLING_LAYOUT_HERON_101: A variant of the Heron layout model.
        DOCLING_LAYOUT_EGRET_MEDIUM: The medium-sized Egret layout model.
        DOCLING_LAYOUT_EGRET_LARGE: The large-sized Egret layout model.
        DOCLING_LAYOUT_EGRET_XLARGE: The extra-large-sized Egret layout model.
    """

    DOCLING_LAYOUT_V2 = "docling_layout_v2"
    DOCLING_LAYOUT_HERON = "docling_layout_heron"
    DOCLING_LAYOUT_HERON_101 = "docling_layout_heron_101"
    DOCLING_LAYOUT_EGRET_MEDIUM = "docling_layout_egret_medium"
    DOCLING_LAYOUT_EGRET_LARGE = "docling_layout_egret_large"
    DOCLING_LAYOUT_EGRET_XLARGE = "docling_layout_egret_xlarge"
    # ALTERNATIVE_LAYOUT = "alternative_layout"
