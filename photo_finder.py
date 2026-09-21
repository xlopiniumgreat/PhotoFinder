"""Local reference-based photo search. See README.md (Russian) for Windows setup."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import uuid

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif'}
LOG = logging.getLogger('photo_finder')
_DLL_HANDLES = []  # Keep Windows DLL search directories registered for this process.


class Cancelled(Exception):
    """Cooperative cancellation between images; reports close normally."""


def check_cancel(stop):
    if stop is not None and stop.is_set():
        raise Cancelled('Обработка остановлена пользователем')


def prepare_cuda_libraries(ort):
    if os.name == 'nt':
        roots = [Path(sys.prefix) / 'Lib' / 'site-packages', Path(getattr(sys, '_MEIPASS', Path(__file__).parent))]
        bins = sorted({p for root in roots for p in (root / 'nvidia').glob('**/bin') if p.is_dir()})
        if bins:
            os.environ['PATH'] = os.pathsep.join(map(str, bins)) + os.pathsep + os.environ.get('PATH', '')
            _DLL_HANDLES.extend(os.add_dll_directory(str(p)) for p in bins)
    if hasattr(ort, 'preload_dlls'):
        ort.preload_dlls(directory='')


def disk_path(path):
    """Extended Windows paths; Pillow receives a file object for Unicode safety."""
    value = str(Path(path).absolute())
    if os.name == 'nt' and not value.startswith('\\\\?\\'):
        return '\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value
    return value


def images(root):
    root = Path(disk_path(root))
    if not root.is_dir():
        raise ValueError(f'Input directory does not exist: {root}')
    def walk_error(error):
        raise OSError(f'Cannot enumerate input directory: {error}')
    result = []
    for directory, subdirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        subdirs[:] = [d for d in subdirs if not (Path(directory) / d).is_symlink()
                      and not getattr(Path(directory) / d, 'is_junction', lambda: False)()]
        result.extend(Path(directory) / f for f in files
                      if Path(f).suffix.lower() in EXTENSIONS and not (Path(directory) / f).is_symlink())
    return sorted(result, key=lambda p: str(p).casefold())


def load_image(path):
    import numpy as np
    from PIL import Image, ImageOps
    from pillow_heif import register_heif_opener
    register_heif_opener()
    with open(disk_path(path), 'rb') as stream:
        with Image.open(stream) as source:
            # Primary frame only (HEIF sequences / animated WebP).
            rgb = ImageOps.exif_transpose(source).convert('RGB')
            return np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])


def normalize(value):
    import numpy as np
    vector = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if vector.ndim != 1 or not np.isfinite(vector).all() or norm <= 0:
        raise ValueError('Invalid face embedding')
    return vector / norm


class Engine:
    def __init__(self, config):
        os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        cuda = config['device'] != 'cpu' and 'CUDAExecutionProvider' in ort.get_available_providers()
        if config['device'] == 'cuda' and not cuda:
            raise RuntimeError('CUDA unavailable; install onnxruntime-gpu or use --device cpu')
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        def create(selected):
            app = FaceAnalysis(name=config['model'], root=str(Path(config['model_dir']).resolve()),
                               allowed_modules=['detection', 'recognition'], providers=selected)
            app.prepare(ctx_id=0 if 'CUDAExecutionProvider' in selected else -1,
                        det_size=(config['det_size'], config['det_size']), det_thresh=config['det_thresh'])
            if 'recognition' not in app.models:
                raise RuntimeError('Model pack contains no recognition model')
            # Prevent ORT from silently switching an active CUDA session to CPU.
            for model in app.models.values():
                model.session.disable_fallback()
            if 'CUDAExecutionProvider' in selected:
                import numpy as np
                for name, model in app.models.items():
                    if 'CUDAExecutionProvider' not in model.session.get_providers():
                        raise RuntimeError(f'{name}: CUDA provider initialization failed')
                    size = config['det_size'] if name == 'detection' else 112
                    model.session.run(None, {model.session.get_inputs()[0].name:
                                             np.zeros((1, 3, size, size), dtype=np.float32)})
            return app
        try:
            if cuda:
                prepare_cuda_libraries(ort)
            self.app = create(providers)
        except Exception:
            if not cuda or config['device'] == 'cuda':
                raise
            LOG.warning('GPU initialization failed; retrying on CPU', exc_info=True)
            self.app = create(['CPUExecutionProvider'])
        self.providers = {name: model.session.get_providers() for name, model in self.app.models.items()}
        if config['device'] == 'cuda' and any('CUDAExecutionProvider' not in p for p in self.providers.values()):
            raise RuntimeError('A model fell back to CPU; repair CUDA dependencies or use --device auto')
        LOG.info('Active providers: %s', self.providers)

    def faces(self, path):
        # InsightFace performs landmark-based alignment before ArcFace inference.
        return self.app.get(load_image(path), max_num=0)


def write_json(path, value):
    with open(disk_path(path), 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def references(engine, root, run, *, paths=None, callback=None, stop=None):
    import numpy as np
    from tqdm import tqdm
    accepted, vectors, records = [], [], []
    selected = images(root) if paths is None else paths
    for number, path in enumerate(tqdm(selected, desc='References', dynamic_ncols=True, ascii=True, disable=callback is not None), 1):
        check_cancel(stop)
        record = {'file': str(path)}
        try:
            faces = engine.faces(path)
            if len(faces) != 1:
                raise ValueError(f'Reference must contain exactly one detected face; found {len(faces)}')
            vectors.append(normalize(faces[0].embedding))
            accepted.append(str(path))
            record['status'] = 'accepted'
        except Exception as error:
            record.update(status='rejected', error=str(error))
            LOG.warning('Reference rejected %s: %s', path, error)
        records.append(record)
        if callback:
            callback(number, len(selected), record)
    write_json(run / 'references.json', records)
    if not vectors:
        raise ValueError('No valid references. See references.json; use photos of yourself alone.')
    if len(vectors) < 15:
        LOG.warning('Only %s valid references; recommend 15–30 diverse solo photos', len(vectors))
    matrix = np.stack(vectors)
    with open(disk_path(run / 'reference_embeddings.npz'), 'wb') as stream:
        np.savez_compressed(stream, embeddings=matrix, files=np.asarray(accepted))
    return matrix, accepted


def classify(score, low, high):
    if score is None:
        return 'uncertain'
    return 'me' if score >= high else 'uncertain' if score >= low else 'not_me'


def analyze(path, root, engine, refs, reference_files, config):
    import numpy as np
    row = dict(file=path.name, relative_path=path.relative_to(root).as_posix(),
               format=path.suffix.lower(), faces_detected=None, me_detected=None,
               best_similarity=None, classification='uncertain', processing_status='ok',
               error=None, faces=[], output_path=None)
    try:
        faces = engine.faces(path)
        row['faces_detected'] = len(faces)
        for face in faces:
            similarities = np.clip(refs @ normalize(face.embedding), -1.0, 1.0)
            index = int(np.argmax(similarities))
            row['faces'].append(dict(similarity=float(similarities[index]),
                                     reference=reference_files[index],
                                     bbox=[float(x) for x in face.bbox],
                                     detection_score=float(face.det_score)))
        score = max((face['similarity'] for face in row['faces']), default=None)
        row.update(best_similarity=score, classification=classify(score, config['t_low'], config['t_high']))
        row['me_detected'] = {'me': True, 'not_me': False, 'uncertain': None}[row['classification']]
        if not faces:
            row['processing_status'] = 'no_faces'
    except Exception as error:
        row.update(processing_status='analysis_error', error=str(error))
        LOG.exception('Cannot analyze %s', path)
    return row


class Report:
    """Streaming JSONL survives interruption; JSON array and CSV also flushed per file."""
    def __init__(self, directory, name):
        self.csv = open(disk_path(directory / f'{name}.csv'), 'w', encoding='utf-8-sig', newline='')
        self.jsonl = open(disk_path(directory / f'{name}.jsonl'), 'w', encoding='utf-8')
        self.json = open(disk_path(directory / f'{name}.json'), 'w', encoding='utf-8')
        self.writer = None
        self.first = True
        self.json.write('[\n')

    def add(self, row):
        serialized = json.dumps(row, ensure_ascii=False, allow_nan=False)
        self.jsonl.write(serialized + '\n')
        self.json.write(('' if self.first else ',\n') + serialized)
        self.first = False
        flat = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()}
        if self.writer is None:
            self.writer = csv.DictWriter(self.csv, fieldnames=list(flat))
            self.writer.writeheader()
        self.writer.writerow(flat)
        for stream in (self.csv, self.jsonl, self.json):
            stream.flush()

    def close(self):
        self.json.write('\n]\n')
        for stream in (self.csv, self.jsonl, self.json):
            stream.close()


def scan(root, engine, refs, names, config, run, name='report', materialize=True, *, callback=None, stop=None):
    from tqdm import tqdm
    root = Path(disk_path(root))
    paths = images(root)
    if not paths:
        raise ValueError(f'No supported images in {root}')
    records, counts = [], Counter()
    report = Report(run, name)
    try:
        progress = tqdm(paths, desc='Processed', unit='photo', dynamic_ncols=True, ascii=True, disable=callback is not None)
        for path in progress:
            check_cancel(stop)
            row = analyze(path, root, engine, refs, names, config)
            if materialize and config['output_mode'] == 'copy':
                target = run / row['classification'] / row['relative_path']
                try:
                    os.makedirs(disk_path(target.parent), exist_ok=True)
                    # New exclusive destination; never replace an existing file.
                    with open(disk_path(path), 'rb') as source, open(disk_path(target), 'xb') as dest:
                        shutil.copyfileobj(source, dest)
                    row['output_path'] = str(target)
                except Exception as error:
                    row['error'] = '; '.join(x for x in (row['error'], f'Copy failed: {error}') if x)
                    row['processing_status'] = 'copy_error'
                    LOG.exception('Cannot copy %s', path)
            report.add(row)
            records.append(row)
            counts[row['classification']] += 1
            if row['processing_status'].endswith('_error'):
                counts['errors'] += 1
            progress.set_postfix(ME=counts['me'], UNCERTAIN=counts['uncertain'],
                                 NOT_ME=counts['not_me'], ERRORS=counts['errors'])
            if callback:
                callback(len(records), len(paths), dict(counts))
    finally:
        report.close()
    LOG.info('Processed: %s / %s | ME: %s | UNCERTAIN: %s | NOT ME: %s | ERRORS: %s',
             len(records), len(paths), counts['me'], counts['uncertain'], counts['not_me'], counts['errors'])
    return records, dict(counts)


def distribution(rows):
    import numpy as np
    scores = [r['best_similarity'] for r in rows if r['processing_status'] == 'ok' and r['best_similarity'] is not None]
    bins = np.linspace(-1, 1, 41)
    return dict(total=len(rows), scored=len(scores), no_faces=sum(r['processing_status'] == 'no_faces' for r in rows),
                errors=sum(r['processing_status'].endswith('_error') for r in rows),
                quantiles=dict(zip(['min', 'p01', 'p05', 'median', 'p95', 'p99', 'max'],
                                   [float(x) for x in np.quantile(scores, [0, .01, .05, .5, .95, .99, 1])])) if scores else {},
                histogram={'edges': bins.tolist(), 'counts': np.histogram(scores, bins=bins)[0].tolist()})


def calibration_metrics(positive, negative, low, high):
    def fractions(rows):
        counts = Counter(classify(r['best_similarity'], low, high) for r in rows)
        return {k: counts[k] / len(rows) for k in ('me', 'uncertain', 'not_me')}
    p, n = fractions(positive), fractions(negative)
    return dict(t_low=low, t_high=high, positive=p, negative=n,
                retained_positive_recall=1 - p['not_me'],
                confident_positive_recall=p['me'], false_positive_me_rate=n['me'])


DEFAULTS = dict(t_low=0.38, t_high=0.50, det_size=1280, det_thresh=0.4,
                model='buffalo_l', model_dir='models', device='cpu', output_mode='copy')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['scan', 'calibrate'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--references', type=Path, default=Path('references'))
    parser.add_argument('--photos', type=Path, default=Path('photos'))
    parser.add_argument('--positive', type=Path, default=Path('positive_samples'))
    parser.add_argument('--negative', type=Path, default=Path('negative_samples'))
    parser.add_argument('--output', type=Path, default=Path('output'))
    for key in ('t_low', 't_high', 'det_thresh'):
        parser.add_argument('--' + key.replace('_', '-'), type=float)
    parser.add_argument('--det-size', type=int)
    parser.add_argument('--model')
    parser.add_argument('--model-dir')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--output-mode', choices=['copy', 'report'])
    args = parser.parse_args(argv)
    config = DEFAULTS.copy()
    if args.config:
        with args.config.open(encoding='utf-8-sig') as stream:
            supplied = json.load(stream)
        if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
            parser.error('Configuration must be an object containing only documented parameters')
        config.update(supplied)
    config.update({k: getattr(args, k) for k in DEFAULTS if getattr(args, k) is not None})
    if not -1 <= config['t_low'] < config['t_high'] <= 1:
        parser.error('Require -1 <= t_low < t_high <= 1')
    if not 0 < config['det_thresh'] <= 1 or not isinstance(config['det_size'], int) or config['det_size'] < 32:
        parser.error('det_thresh must be in (0, 1]; det_size integer >= 32')
    if config['device'] not in ('auto', 'cpu', 'cuda') or config['output_mode'] not in ('copy', 'report'):
        parser.error('Invalid device or output_mode in configuration')
    roots = [args.references, args.photos] if args.command == 'scan' else [args.references, args.positive, args.negative]
    resolved = [p.resolve() for p in roots]
    destinations = [args.output.resolve(), Path(config['model_dir']).resolve()]
    for index, source in enumerate(resolved):
        if not source.is_dir():
            parser.error(f'Input directory does not exist: {source}')
        for other in resolved[index + 1:] + destinations:
            if source.is_relative_to(other) or other.is_relative_to(source):
                parser.error(f'Directories must not overlap: {source} and {other}')
    run = args.output.resolve() / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    os.makedirs(disk_path(run), exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(disk_path(run / 'run.log'), encoding='utf-8')])
    LOG.info('Output: %s', run)
    write_json(run / 'config.json', config)
    engine = Engine(config)
    write_json(run / 'providers.json', engine.providers)
    refs, names = references(engine, args.references, run)
    if args.command == 'scan':
        for category in ('me', 'uncertain', 'not_me'):
            os.makedirs(disk_path(run / category), exist_ok=True)
        _, counts = scan(args.photos, engine, refs, names, config, run)
        write_json(run / 'summary.json', counts)
        return 2 if counts.get('errors') else 0
    positive, _ = scan(args.positive, engine, refs, names, config, run, 'positive', False)
    negative, _ = scan(args.negative, engine, refs, names, config, run, 'negative', False)
    # No automatic threshold claim: export distributions and empirical operating points.
    result = {'same_person': distribution(positive), 'different_person': distribution(negative),
              'current_thresholds': calibration_metrics(positive, negative, config['t_low'], config['t_high']),
              'threshold_grid': [calibration_metrics(positive, negative, low / 100, high / 100)
                                 for low in range(20, 61, 2) for high in range(low + 2, 81, 2)]}
    write_json(run / 'calibration.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'threshold_grid'}, ensure_ascii=False, indent=2))
    return 2 if result['same_person']['errors'] or result['different_person']['errors'] else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted. Completed records remain in CSV/JSONL/JSON.', file=sys.stderr)
        sys.exit(130)
    except Exception:
        logging.exception('Run failed')
        sys.exit(1)
