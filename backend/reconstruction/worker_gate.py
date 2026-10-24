"""Bound local reconstruction concurrency in the single-process desktop server."""
from functools import wraps
from threading import Lock

_worker_lock=Lock()

def serialized_reconstruction(function):
    @wraps(function)
    def run(*args,**kwargs):
        # Jobs retain distinct IDs and remain queued until the active worker
        # finishes. Always release on failures; never share output directories.
        with _worker_lock:
            return function(*args,**kwargs)
    return run
