"""GUI-independent jobs. Only plain data crosses the UI/worker boundary."""
from collections import Counter
from datetime import datetime
import hashlib
import logging
from pathlib import Path
import shutil
import sys
import uuid

import photo_finder as core


def prepare_bundled_models(config):
    source = Path(getattr(sys, '_MEIPASS', Path(__file__).parent)) / 'bundled_models' / config['model']
    if source.is_dir():
        target = Path(config['model_dir']) / 'models' / config['model']
        target.mkdir(parents=True, exist_ok=True)
        for file in source.glob('*.onnx'):
            dest = target / file.name
            if not dest.exists() or dest.stat().st_size != file.stat().st_size:
                temp = target / (file.name + '.partial')
                shutil.copyfile(file, temp)
                temp.replace(dest)


def validate(job):
    groups = [('references', 1, 'Эталоны'), ('positive', 3, 'Фото с человеком'),
              ('negative', 3, 'Фото без человека')]
    seen = set()
    for key, minimum, label in groups:
        files = job[key]
        if len(files) < minimum:
            raise ValueError(f'{label}: выберите минимум {minimum} фото.')
        for file in files:
            p = Path(file).resolve()
            if not p.is_file() or p.suffix.lower() not in core.EXTENSIONS:
                raise ValueError(f'Не удалось прочитать файл: {p}')
            if p in seen:
                raise ValueError('Один снимок нельзя использовать одновременно в эталонах и проверочных наборах.')
            seen.add(p)
    low, high = job['config']['t_low'], job['config']['t_high']
    if not -1 <= low < high <= 1:
        raise ValueError('Пороги должны удовлетворять: -1 ≤ нижний < верхний ≤ 1.')
    output = Path(job['output']).resolve()
    for file in seen:
        if file.is_relative_to(output):
            raise ValueError('Выберите папку результатов, не содержащую исходные фотографии.')
    model = Path(job['config']['model_dir']).resolve()
    if output.is_relative_to(model) or model.is_relative_to(output):
        raise ValueError('Папки моделей и результатов должны быть раздельными.')
    if job['mode'] == 'scan':
        photos = Path(job['photos']).resolve()
        if not photos.is_dir():
            raise ValueError('Выберите существующую папку с фотографиями для поиска.')
        if any(photos.is_relative_to(p) or p.is_relative_to(photos) for p in (output, model)):
            raise ValueError('Папки поиска, моделей и результатов не должны находиться друг внутри друга.')


def run_job(job, emit, stop, engine_factory=core.Engine):
    validate(job)
    # Reject renamed copies as well as identical paths in verification datasets.
    hashes = set()
    for key in ('references', 'positive', 'negative'):
        for file in job[key]:
            core.check_cancel(stop)
            with open(core.disk_path(file), 'rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').digest()
            if digest in hashes:
                raise ValueError('Найдены одинаковые файлы в выбранных фото. Уберите дубликаты, в том числе переименованные копии.')
            hashes.add(digest)
    run = Path(job['output']).resolve() / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    emit('output', str(run))
    core.write_json(run / 'config.json', job['config'])
    core.write_json(run / 'inputs.json', {k: job[k] for k in ('references', 'positive', 'negative', 'photos')})
    handler = logging.FileHandler(run / 'run.log', encoding='utf-8')
    core.LOG.addHandler(handler)
    core.LOG.setLevel(logging.INFO)
    try:
        emit('stage', 'Загрузка модели. При первом запуске требуется интернет…')
        prepare_bundled_models(job['config'])
        engine = engine_factory(job['config'])
        core.write_json(run / 'providers.json', engine.providers)
        core.check_cancel(stop)
        emit('stage', 'Проверка эталонных фотографий')
        refs, names = core.references(engine, None, run, paths=[Path(p) for p in job['references']], stop=stop,
            callback=lambda n, total, record: emit('progress', (n, total, f'Эталоны: {n} / {total}')))
        emit('reference_count', (len(names), len(job['references'])))
        samples = {}
        for key, label in [('positive', 'Фото с человеком'), ('negative', 'Фото без человека')]:
            emit('stage', f'Проверочные примеры: {label.lower()}')
            rows = []
            report = core.Report(run, key)
            try:
                for n, file in enumerate(job[key], 1):
                    core.check_cancel(stop)
                    p = Path(file)
                    row = core.analyze(p, p.parent, engine, refs, names, job['config'])
                    row['source_path'] = str(p)
                    report.add(row)
                    rows.append(row)
                    emit('progress', (n, len(job[key]), f'{label}: {n} / {len(job[key])}'))
            finally:
                report.close()
            samples[key] = rows
        result = dict(metrics=core.calibration_metrics(samples['positive'], samples['negative'],
                     job['config']['t_low'], job['config']['t_high']),
                     positive=dict(Counter(r['classification'] for r in samples['positive'])),
                     negative=dict(Counter(r['classification'] for r in samples['negative'])),
                     errors=sum(r['processing_status'].endswith('_error') for rows in samples.values() for r in rows))
        core.write_json(run / 'sample_check.json', result)
        emit('sample_result', result)
        if result['errors'] or result['positive'].get('not_me', 0) or result['negative'].get('me', 0):
            details = []
            missed = result['positive'].get('not_me', 0)
            false_matches = result['negative'].get('me', 0)
            if missed:
                details.append(
                    f'На {missed} из {len(samples["positive"])} фото с человеком он не распознан '
                    '(категория not_me). В основной коллекции тоже возможны пропуски; '
                    'проверьте эти снимки в positive.csv.')
            if false_matches:
                details.append(
                    f'На {false_matches} из {len(samples["negative"])} фото без человека '
                    'найдено ложное совпадение (категория me). '
                    'Проверьте эти снимки в negative.csv.')
            if result['errors']:
                details.append(
                    f'Не удалось прочитать или обработать проверочных фото: {result["errors"]}. '
                    'Причины указаны в поле error отчётов positive.csv и negative.csv.')
            details.append('Отчёты доступны через «Открыть результаты». '
                           'Пропуски и ложные совпадения не входят в счётчик технических ошибок.')
            details.append('Поиск по основной папке продолжится с текущими порогами.' if job['mode'] == 'scan' else
                           'Можно нажать «Начать поиск»: эти результаты не блокируют запуск. Пороги не изменены.')
            emit('sample_warning', '\n'.join(details))
        if job['mode'] == 'check':
            return
        core.check_cancel(stop)
        for category in ('me', 'uncertain', 'not_me'):
            (run / category).mkdir()
        emit('stage', 'Поиск фотографий')
        rows, counts = core.scan(Path(job['photos']), engine, refs, names, job['config'], run, stop=stop,
            callback=lambda n, total, counts: emit('scan_progress', (n, total, counts)))
        core.write_json(run / 'summary.json', counts)
        emit('summary', counts)
    except core.Cancelled:
        core.write_json(run / 'status.json', {'status': 'cancelled'})
        raise
    except Exception:
        core.LOG.exception('GUI job failed')
        core.write_json(run / 'status.json', {'status': 'error'})
        raise
    finally:
        core.LOG.removeHandler(handler)
        handler.close()
