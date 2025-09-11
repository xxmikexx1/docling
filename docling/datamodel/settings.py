import sys
from pathlib import Path
from typing import Annotated, Optional, Tuple

from pydantic import BaseModel, PlainValidator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _validate_page_range(v: Tuple[int, int]) -> Tuple[int, int]:
    if v[0] < 1 or v[1] < v[0]:
        raise ValueError(
            "Invalid page range: start must be ≥ 1 and end must be ≥ start."
        )
    return v


PageRange = Annotated[Tuple[int, int], PlainValidator(_validate_page_range)]

DEFAULT_PAGE_RANGE: PageRange = (1, sys.maxsize)


class DocumentLimits(BaseModel):
    """Defines the processing limits for a single document.

    This class specifies constraints on the size and page range of documents
    to be processed, preventing excessive resource usage.

    Attributes:
        max_num_pages: The maximum number of pages a document can have.
        max_file_size: The maximum file size in bytes.
        page_range: A tuple specifying the start and end pages to process.
    """

    max_num_pages: int = sys.maxsize
    max_file_size: int = sys.maxsize
    page_range: PageRange = DEFAULT_PAGE_RANGE


class BatchConcurrencySettings(BaseModel):
    """Defines settings for batching and concurrency during processing.

    This class controls how documents, pages, and elements are grouped into
    batches and processed concurrently, allowing for performance tuning.

    Attributes:
        doc_batch_size: The number of documents to process in a single batch.
        doc_batch_concurrency: The number of parallel threads for processing
            document batches.
        page_batch_size: The number of pages to process in a single batch.
        page_batch_concurrency: The number of parallel threads for processing
            page batches (currently unused).
        elements_batch_size: The number of elements to process in a single batch
            for enrichment models.
    """

    doc_batch_size: int = 1  # Number of documents processed in one batch. Should be >= doc_batch_concurrency
    doc_batch_concurrency: int = 1  # Number of parallel threads processing documents. Warning: Experimental! No benefit expected without free-threaded python.
    page_batch_size: int = 4  # Number of pages processed in one batch.
    page_batch_concurrency: int = 1  # Currently unused.
    elements_batch_size: int = (
        16  # Number of elements processed in one batch, in enrichment models.
    )

    # To force models into single core: export OMP_NUM_THREADS=1


class DebugSettings(BaseModel):
    """Defines settings for enabling and controlling debug visualizations.

    This class provides a set of boolean flags to enable various debugging
    outputs, such as visualizations of cell detection, OCR results, and layout
    analysis. It also includes settings for performance profiling.

    Attributes:
        visualize_cells: If `True`, generates visualizations of detected text cells.
        visualize_ocr: If `True`, generates visualizations of OCR results.
        visualize_layout: If `True`, generates visualizations of the final
            layout analysis.
        visualize_raw_layout: If `True`, generates visualizations of the raw
            output from the layout model.
        visualize_tables: If `True`, generates visualizations of detected tables.
        profile_pipeline_timings: If `True`, enables profiling of pipeline
            execution times.
        debug_output_path: The path to the directory where debug artifacts
            should be saved.
    """

    visualize_cells: bool = False
    visualize_ocr: bool = False
    visualize_layout: bool = False
    visualize_raw_layout: bool = False
    visualize_tables: bool = False

    profile_pipeline_timings: bool = False

    # Path used to output debug information.
    debug_output_path: str = str(Path.cwd() / "debug")


class AppSettings(BaseSettings):
    """The main application settings model for Docling.

    This class aggregates all the major settings groups for the application,
    including performance, debugging, and cache directory. It uses
    `pydantic-settings` to load configuration from environment variables
    with the prefix `DOCLING_`.

    Attributes:
        perf: A `BatchConcurrencySettings` object that controls performance-related
            settings like batching and concurrency.
        debug: A `DebugSettings` object that controls debugging features like
            visualizations and profiling.
        cache_dir: The path to the directory used for caching models and other
            data.
        artifacts_path: An optional path to a directory where output artifacts
            are stored.
    """

    model_config = SettingsConfigDict(
        env_prefix="DOCLING_", env_nested_delimiter="_", env_nested_max_split=1
    )

    perf: BatchConcurrencySettings = BatchConcurrencySettings()
    debug: DebugSettings = DebugSettings()

    cache_dir: Path = Path.home() / ".cache" / "docling"
    artifacts_path: Optional[Path] = None


settings = AppSettings()
