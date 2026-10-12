"""Create a local SMP 0.3.2 copy with the FiDi model files.

Source and replacement hashes are checked before copying. The installed
package is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "smp-0.3.2"
OUTPUT = ROOT / ".local" / "segmentation_models_pytorch"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="SMP 0.3.2 package directory; defaults to installed package")
    args = parser.parse_args()

    if args.source:
        source = args.source.resolve()
    else:
        spec = importlib.util.find_spec("segmentation_models_pytorch")
        if spec is None or not spec.submodule_search_locations:
            raise SystemExit("Install segmentation-models-pytorch==0.3.2 or pass --source")
        source = Path(next(iter(spec.submodule_search_locations))).resolve()

    if source == OUTPUT.resolve():
        raise SystemExit("Source must be the unmodified SMP package, not .local")
    if not (source / "__version__.py").exists():
        raise SystemExit(f"Not an SMP source directory: {source}")
    version = (source / "__version__.py").read_text()
    if "VERSION = (0, 3, 2)" not in version:
        raise SystemExit("Expected SMP version 0.3.2")

    manifest = json.loads((PATCH / "manifest.json").read_text())
    for name, hashes in manifest.items():
        upstream = source / name
        accepted = {hashes["base_sha256"]}
        if hashes.get("pypi_sha256"):
            accepted.add(hashes["pypi_sha256"])
        if digest(upstream) not in accepted:
            raise SystemExit(f"Upstream file is not a recognized SMP 0.3.2 source: {name}")
        if digest(PATCH / name) != hashes["modified_sha256"]:
            raise SystemExit(f"FiDi replacement hash mismatch: {name}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    shutil.copytree(source, OUTPUT, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in manifest:
        shutil.copyfile(PATCH / name, OUTPUT / name)
    print(f"Prepared FiDi model package at {OUTPUT}")


if __name__ == "__main__":
    main()
