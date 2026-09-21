"""Create GitHub-size portable release archives from a PyInstaller onedir build."""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

GITHUB_ASSET_LIMIT = 2_000_000_000  # Keep each asset below GitHub's 2 GiB limit.


def write_zip(root: Path, files: list[Path], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for file in files:
            archive.write(file, file.relative_to(root).as_posix())
    size = destination.stat().st_size
    if size >= GITHUB_ASSET_LIMIT:
        destination.unlink(missing_ok=True)
        raise SystemExit(f"{destination.name} exceeds GitHub's per-asset limit ({size:,} bytes)")
    print(f"{destination.name}: {size:,} bytes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=Path, required=True, help="PyInstaller dist/PhotoFinder folder")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    app = args.app.resolve()
    internal = app / "_internal"
    nvidia = internal / "nvidia"
    if not (app / "PhotoFinder.exe").is_file() or not internal.is_dir() or not nvidia.is_dir():
        parser.error("expected PhotoFinder.exe and _internal/nvidia in the built app folder")

    base_files = [p for p in app.rglob("*") if p.is_file() and nvidia not in p.parents]
    cudnn_files = [p for p in (nvidia / "cudnn").rglob("*") if p.is_file()]
    cuda_files = [p for p in nvidia.rglob("*") if p.is_file() and (nvidia / "cudnn") not in p.parents]
    if not cudnn_files or not cuda_files:
        parser.error("NVIDIA runtime must contain both cudnn and other CUDA libraries")
    write_zip(app, base_files, args.out / "PhotoFinder-Windows-CPU.zip")
    write_zip(app, cudnn_files, args.out / "PhotoFinder-NVIDIA-1-CuDNN.zip")
    write_zip(app, cuda_files, args.out / "PhotoFinder-NVIDIA-2-CUDA.zip")


if __name__ == "__main__":
    main()
