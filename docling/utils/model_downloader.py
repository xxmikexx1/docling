import logging
from pathlib import Path
from typing import Optional

from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_V2
from docling.datamodel.pipeline_options import (
    LayoutOptions,
    granite_picture_description,
    smolvlm_picture_description,
)
from docling.datamodel.settings import settings
from docling.datamodel.vlm_model_specs import (
    SMOLDOCLING_MLX,
    SMOLDOCLING_TRANSFORMERS,
)
from docling.models.code_formula_model import CodeFormulaModel
from docling.models.document_picture_classifier import DocumentPictureClassifier
from docling.models.easyocr_model import EasyOcrModel
from docling.models.layout_model import LayoutModel
from docling.models.picture_description_vlm_model import PictureDescriptionVlmModel
from docling.models.table_structure_model import TableStructureModel
from docling.models.utils.hf_model_download import download_hf_model

_log = logging.getLogger(__name__)


def download_models(
    output_dir: Optional[Path] = None,
    *,
    force: bool = False,
    progress: bool = False,
    with_layout: bool = True,
    with_tableformer: bool = True,
    with_code_formula: bool = True,
    with_picture_classifier: bool = True,
    with_smolvlm: bool = False,
    with_smoldocling: bool = False,
    with_smoldocling_mlx: bool = False,
    with_granite_vision: bool = False,
    with_easyocr: bool = True,
):
    """Downloads all the required machine learning models for Docling.

    This utility function provides a convenient way to pre-download and cache all
    the models used in the various Docling pipelines. This is useful for
    setting up an environment, especially in offline or containerized settings.

    Args:
        output_dir: The directory where the models should be saved. If not
            provided, it defaults to the cache directory specified in the
            application settings.
        force: If `True`, forces the re-download of models even if they
            already exist in the cache.
        progress: If `True`, displays a progress bar during the download.
        with_layout: If `True`, downloads the layout analysis model.
        with_tableformer: If `True`, downloads the table structure model.
        with_code_formula: If `True`, downloads the code and formula detection model.
        with_picture_classifier: If `True`, downloads the picture classification model.
        with_smolvlm: If `True`, downloads the SmolVLM model.
        with_smoldocling: If `True`, downloads the SmolDocling model.
        with_smoldocling_mlx: If `True`, downloads the MLX version of the
            SmolDocling model.
        with_granite_vision: If `True`, downloads the Granite Vision model.
        with_easyocr: If `True`, downloads the EasyOCR models.

    Returns:
        The path to the directory where the models were downloaded.
    """
    if output_dir is None:
        output_dir = settings.cache_dir / "models"

    # Make sure the folder exists
    output_dir.mkdir(exist_ok=True, parents=True)

    if with_layout:
        _log.info("Downloading layout model...")
        LayoutModel.download_models(
            local_dir=output_dir / LayoutOptions().model_spec.model_repo_folder,
            force=force,
            progress=progress,
        )

    if with_tableformer:
        _log.info("Downloading tableformer model...")
        TableStructureModel.download_models(
            local_dir=output_dir / TableStructureModel._model_repo_folder,
            force=force,
            progress=progress,
        )

    if with_picture_classifier:
        _log.info("Downloading picture classifier model...")
        DocumentPictureClassifier.download_models(
            local_dir=output_dir / DocumentPictureClassifier._model_repo_folder,
            force=force,
            progress=progress,
        )

    if with_code_formula:
        _log.info("Downloading code formula model...")
        CodeFormulaModel.download_models(
            local_dir=output_dir / CodeFormulaModel._model_repo_folder,
            force=force,
            progress=progress,
        )

    if with_smolvlm:
        _log.info("Downloading SmolVlm model...")
        download_hf_model(
            repo_id=smolvlm_picture_description.repo_id,
            local_dir=output_dir / smolvlm_picture_description.repo_cache_folder,
            force=force,
            progress=progress,
        )

    if with_smoldocling:
        _log.info("Downloading SmolDocling model...")
        download_hf_model(
            repo_id=SMOLDOCLING_TRANSFORMERS.repo_id,
            local_dir=output_dir / SMOLDOCLING_TRANSFORMERS.repo_cache_folder,
            force=force,
            progress=progress,
        )

    if with_smoldocling_mlx:
        _log.info("Downloading SmolDocling MLX model...")
        download_hf_model(
            repo_id=SMOLDOCLING_MLX.repo_id,
            local_dir=output_dir / SMOLDOCLING_MLX.repo_cache_folder,
            force=force,
            progress=progress,
        )

    if with_granite_vision:
        _log.info("Downloading Granite Vision model...")
        download_hf_model(
            repo_id=granite_picture_description.repo_id,
            local_dir=output_dir / granite_picture_description.repo_cache_folder,
            force=force,
            progress=progress,
        )

    if with_easyocr:
        _log.info("Downloading easyocr models...")
        EasyOcrModel.download_models(
            local_dir=output_dir / EasyOcrModel._model_repo_folder,
            force=force,
            progress=progress,
        )

    return output_dir
