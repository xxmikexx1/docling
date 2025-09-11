import logging
import os
import re
from enum import Enum
from typing import Any, Union

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_log = logging.getLogger(__name__)


class AcceleratorDevice(str, Enum):
    """An enumeration of supported accelerator devices for model inference.

    This enum defines the types of hardware accelerators that can be used to run
    machine learning models, providing a standardized way to specify the execution
    device.

    Attributes:
        AUTO: Automatically selects the best available device.
        CPU: Uses the central processing unit for computation.
        CUDA: Uses a NVIDIA CUDA-enabled graphics processing unit (GPU).
        MPS: Uses the Metal Performance Shaders (MPS) framework on Apple silicon.
    """

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class AcceleratorOptions(BaseSettings):
    """Manages configuration settings for hardware acceleration.

    This class defines the settings for controlling how machine learning models are
    executed, including the number of threads, the target device, and device-specific
    options like Flash Attention 2 for CUDA. It uses `pydantic-settings` to load
    configuration from environment variables.

    Attributes:
        num_threads: The number of threads to use for processing. Defaults to 4.
        device: The hardware device for model inference. Can be specified as a string
            (e.g., "auto", "cpu", "cuda", "cuda:0", "mps") or as an `AcceleratorDevice`
            enum member. Defaults to "auto".
        cuda_use_flash_attention2: A boolean flag to enable or disable Flash Attention 2
            for CUDA devices, which can improve performance. Defaults to False.
    """

    model_config = SettingsConfigDict(
        env_prefix="DOCLING_", env_nested_delimiter="_", populate_by_name=True
    )

    num_threads: int = 4
    device: Union[str, AcceleratorDevice] = "auto"
    cuda_use_flash_attention2: bool = False

    @field_validator("device")
    def validate_device(cls, value):
        """Validates the 'device' field.

        This validator ensures that the specified device is a valid choice.
        It accepts standard device names like "auto", "cpu", "mps", "cuda",
        or a CUDA device with a specific ID (e.g., "cuda:0").

        Args:
            value: The device string to validate.

        Returns:
            The validated device string.

        Raises:
            ValueError: If the device string is not a valid option.
        """
        # "auto", "cpu", "cuda", "mps", or "cuda:N"
        if value in {d.value for d in AcceleratorDevice} or re.match(
            r"^cuda(:\d+)?$", value
        ):
            return value
        raise ValueError(
            "Invalid device option. Use 'auto', 'cpu', 'mps', 'cuda', or 'cuda:N'."
        )

    @model_validator(mode="before")
    @classmethod
    def check_alternative_envvars(cls, data: Any) -> Any:
        """Sets 'num_threads' from an alternative environment variable.

        This model validator allows setting the `num_threads` attribute from the
        `OMP_NUM_THREADS` environment variable as a fallback if `DOCLING_NUM_THREADS`
        is not defined. This is useful for maintaining compatibility with existing
        environment configurations.

        The alternative environment variable is only used if it is valid and the
        primary one is not set. This approach is used because Pydantic's standard
        "aliases" feature does not handle this specific override behavior correctly.

        Args:
            data: The dictionary of raw configuration values.

        Returns:
            The (potentially modified) dictionary of configuration values.
        """
        if isinstance(data, dict):
            input_num_threads = data.get("num_threads")
            # Check if to set the num_threads from the alternative envvar
            if input_num_threads is None:
                docling_num_threads = os.getenv("DOCLING_NUM_THREADS")
                omp_num_threads = os.getenv("OMP_NUM_THREADS")
                if docling_num_threads is None and omp_num_threads is not None:
                    try:
                        data["num_threads"] = int(omp_num_threads)
                    except ValueError:
                        _log.error(
                            "Ignoring misformatted envvar OMP_NUM_THREADS '%s'",
                            omp_num_threads,
                        )
        return data
