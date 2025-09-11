import hashlib
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import List, Union

import requests
from tqdm import tqdm


def chunkify(iterator, chunk_size):
    """Yields successive chunks of a specified size from an iterable.

    This function takes an iterable and divides it into chunks of a given size,
    which is useful for batch processing.

    Args:
        iterator: The iterable to be chunked.
        chunk_size: The desired size of each chunk.

    Yields:
        A list representing a chunk of the original iterable.
    """
    if isinstance(iterator, List):
        iterator = iter(iterator)
    for first in iterator:  # Take the first element from the iterator
        yield [first, *list(islice(iterator, chunk_size - 1))]


def create_file_hash(path_or_stream: Union[BytesIO, Path]) -> str:
    """Creates a SHA-256 hash of a file's content.

    This function generates a stable hash for a file, whether it is provided as
    a local file path or an in-memory stream. It reads the file in chunks to
    handle large files efficiently.

    Args:
        path_or_stream: The source of the file, either a `Path` object or a
            `BytesIO` stream.

    Returns:
        A string containing the hexadecimal representation of the SHA-256 hash.
    """

    block_size = 65536
    hasher = hashlib.sha256(usedforsecurity=False)

    def _hash_buf(binary_stream):
        buf = binary_stream.read(block_size)  # read and page_hash in chunks
        while len(buf) > 0:
            hasher.update(buf)
            buf = binary_stream.read(block_size)

    if isinstance(path_or_stream, Path):
        with path_or_stream.open("rb") as afile:
            _hash_buf(afile)
    elif isinstance(path_or_stream, BytesIO):
        _hash_buf(path_or_stream)

    return hasher.hexdigest()


def create_hash(string: str):
    """Creates a SHA-256 hash of a string.

    Args:
        string: The input string to be hashed.

    Returns:
        A string containing the hexadecimal representation of the SHA-256 hash.
    """
    hasher = hashlib.sha256(usedforsecurity=False)
    hasher.update(string.encode("utf-8"))

    return hasher.hexdigest()


def download_url_with_progress(url: str, progress: bool = False) -> BytesIO:
    """Downloads a file from a URL with an optional progress bar.

    This function streams the content from a given URL into an in-memory
    `BytesIO` buffer. It can display a progress bar using `tqdm` if requested.

    Args:
        url: The URL of the file to download.
        progress: If `True`, displays a progress bar during the download.

    Returns:
        A `BytesIO` object containing the downloaded file content.
    """
    buf = BytesIO()
    with requests.get(url, stream=True, allow_redirects=True) as response:
        total_size = int(response.headers.get("content-length", 0))
        progress_bar = tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            disable=(not progress),
        )

        for chunk in response.iter_content(10 * 1024):
            buf.write(chunk)
            progress_bar.update(len(chunk))
        progress_bar.close()

    buf.seek(0)
    return buf
