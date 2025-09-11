import time
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, List

import numpy as np
from pydantic import BaseModel

from docling.datamodel.settings import settings

if TYPE_CHECKING:
    from docling.datamodel.document import ConversionResult


class ProfilingScope(str, Enum):
    """An enumeration for the scope of a profiling measurement.

    This enum distinguishes between performance metrics that are collected on a
    per-page basis versus those that apply to the entire document.

    Attributes:
        PAGE: The profiling item applies to a single page.
        DOCUMENT: The profiling item applies to the whole document.
    """

    PAGE = "page"
    DOCUMENT = "document"


class ProfilingItem(BaseModel):
    """Represents a collection of timing measurements for a specific operation.

    This class stores a series of timing data for a profiled section of code,
    including the scope, count, and individual time measurements. It also provides
    methods for calculating basic statistics on the collected data.

    Attributes:
        scope: The `ProfilingScope` of the measurements (e.g., "page" or "document").
        count: The number of times the operation was measured.
        times: A list of the elapsed time in seconds for each measurement.
        start_timestamps: A list of `datetime` objects indicating when each
            measurement started.
    """

    scope: ProfilingScope
    count: int = 0
    times: List[float] = []
    start_timestamps: List[datetime] = []

    def avg(self) -> float:
        """Calculates the average of the collected times."""
        return np.average(self.times)  # type: ignore

    def std(self) -> float:
        """Calculates the standard deviation of the collected times."""
        return np.std(self.times)  # type: ignore

    def mean(self) -> float:
        """Calculates the mean of the collected times."""
        return np.mean(self.times)  # type: ignore

    def percentile(self, perc: float) -> float:
        """Calculates a given percentile of the collected times.

        Args:
            perc: The percentile to calculate (between 0 and 100).

        Returns:
            The calculated percentile value.
        """
        return np.percentile(self.times, perc)  # type: ignore


class TimeRecorder:
    """A context manager for recording the execution time of a block of code.

    This class provides a convenient way to profile a section of code using a
    `with` statement. It records the start and end times and appends the
    elapsed time to the appropriate `ProfilingItem` in a `ConversionResult`.

    Attributes:
        conv_res: The `ConversionResult` object where the timing data is stored.
        key: The key to identify this specific measurement in the `timings`
            dictionary of the `ConversionResult`.
    """

    def __init__(
        self,
        conv_res: "ConversionResult",
        key: str,
        scope: ProfilingScope = ProfilingScope.PAGE,
    ):
        """Initializes the TimeRecorder.

        Args:
            conv_res: The `ConversionResult` object to store the profiling data.
            key: The name of the profiling measurement.
            scope: The `ProfilingScope` of the measurement.
        """
        if settings.debug.profile_pipeline_timings:
            if key not in conv_res.timings.keys():
                conv_res.timings[key] = ProfilingItem(scope=scope)
            self.conv_res = conv_res
            self.key = key

    def __enter__(self):
        if settings.debug.profile_pipeline_timings:
            self.start = time.monotonic()
            self.conv_res.timings[self.key].start_timestamps.append(datetime.utcnow())
        return self

    def __exit__(self, *args):
        if settings.debug.profile_pipeline_timings:
            elapsed = time.monotonic() - self.start
            self.conv_res.timings[self.key].times.append(elapsed)
            self.conv_res.timings[self.key].count += 1
