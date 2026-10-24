"""OS-owned cross-process locks, released even if a worker is terminated."""
import os
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def process_lock(path, timeout=120, waiting=None):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    stream=open(path,'a+b')
    locked=False
    try:
        stream.seek(0,2)
        if stream.tell()==0:
            stream.write(b'0'); stream.flush()
        started=time.monotonic()
        while not locked:
            try:
                stream.seek(0)
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                locked=True
            except (OSError,BlockingIOError):
                if time.monotonic()-started>=timeout:
                    raise TimeoutError('Timed out waiting for shared process lock')
                if waiting: waiting()
                time.sleep(.1)
        yield
    finally:
        if locked:
            stream.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(stream,fcntl.LOCK_UN)
        stream.close()
