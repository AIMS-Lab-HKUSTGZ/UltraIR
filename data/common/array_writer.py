"""Bounded-memory NumPy output, including filesystems without mmap support."""
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile

import numpy as np
from numpy.lib.format import open_memmap


@contextmanager
def array_writer(path: Path, shape: tuple, dtype=np.float32):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with tempfile.TemporaryDirectory(prefix='ultrair-array-') as scratch:
        try:
            array = open_memmap(temporary, mode='w+', shape=shape, dtype=dtype)
            fallback = False
        except OSError as exc:
            if exc.errno not in (19, 22, 95):
                raise
            local = Path(scratch) / path.name
            array = open_memmap(local, mode='w+', shape=shape, dtype=dtype)
            fallback = True
        try:
            yield array
            array.flush()
            del array
            if fallback:
                shutil.copyfile(local, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
