"""Build a portable folder with Python, libraries, Tk and optional local models."""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, help='Existing buffalo_l folder; optionally bundle models for offline first launch')
    parser.add_argument('--dist', type=Path, default=Path('dist'))
    parser.add_argument('--work', type=Path, default=Path('build'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    args.dist = args.dist.resolve()
    args.work = args.work.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--onedir', '--windowed',
               '--name', 'PhotoFinder', '--noupx', '--distpath', str(args.dist),
               '--workpath', str(args.work / 'cache'), '--specpath', str(args.work)]
    for package in ('insightface', 'onnxruntime', 'pillow_heif', 'skimage', 'albumentations'):
        command.extend(['--collect-all', package])
    for name in ('onnxruntime-gpu', 'onnxruntime', 'nvidia-cudnn-cu12'):
        from importlib.metadata import PackageNotFoundError, version
        try:
            version(name)
            command.extend(['--copy-metadata', name])
        except PackageNotFoundError:
            pass
    # Preserve wheel DLL layout so prepare_cuda_libraries also works when frozen.
    nvidia = Path(sys.prefix) / 'Lib' / 'site-packages' / 'nvidia'
    if nvidia.is_dir():
        command.extend(['--add-data', f'{nvidia};nvidia'])
    if args.models:
        models = args.models.resolve()
        if not all((models / name).is_file() for name in ('det_10g.onnx', 'w600k_r50.onnx')):
            parser.error('--models must point to a buffalo_l folder containing both required ONNX models')
        command.extend(['--add-data', f'{models};bundled_models/buffalo_l'])
    command.append(str(root / 'photo_finder_gui.py'))
    subprocess.run(command, check=True)
    shutil.copy2(root / 'GUI_README.md', args.dist / 'PhotoFinder' / 'README.md')
    print('Ready:', args.dist / 'PhotoFinder' / 'PhotoFinder.exe')


if __name__ == '__main__':
    main()
