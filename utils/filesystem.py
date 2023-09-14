import os
import zipfile
from pathlib import Path


def mkdir(directory: str, exist_ok: bool = False):
    Path(directory).mkdir(parents=True, exist_ok=exist_ok)


def parent(path: str) -> Path:
    return Path(path).parent.absolute()


def unzip(file_path: str, remove: bool = False):
    archive = zipfile.ZipFile(file_path)
    archive.extractall(parent(file_path))
    archive.close()

    if remove:
        os.remove(file_path)
