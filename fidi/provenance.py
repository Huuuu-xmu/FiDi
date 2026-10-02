"""Record the environment and code identity without collecting clinical data."""

from __future__ import annotations

import hashlib
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from fidi import __version__


def execution_metadata() -> dict:
    packages = {}
    for name in (
        "torch", "torchvision", "segmentation-models-pytorch", "timm",
        "numpy", "Pillow", "scipy", "opencv-python-headless",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).resolve().parents[1]
    fingerprint = hashlib.sha256()
    for directory in ("fidi", "scripts", "patches"):
        for path in sorted((root / directory).rglob("*.py")):
            fingerprint.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
            fingerprint.update(path.read_bytes())
    return {
        "code_version": __version__,
        "source_sha256": fingerprint.hexdigest(),
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": packages,
    }
