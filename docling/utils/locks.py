import threading

"""
This module provides thread locks for managing concurrent access to shared
resources within the Docling library.
"""

pypdfium2_lock = threading.Lock()
"""A thread lock for synchronizing access to the pypdfium2 library.

This lock is used to prevent race conditions and ensure thread safety when
multiple threads are interacting with the pypdfium2 PDF processor, which may
not be inherently thread-safe in all contexts.
"""
