import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

import numpy as np
import photo_finder as core
from gui_jobs import run_job, validate


class FakeEngine:
    providers = {'detection': ['CPUExecutionProvider'], 'recognition': ['CPUExecutionProvider']}
    def __init__(self, config):
        pass
    def faces(self, path):
        vector = [0, 1] if Path(path).name.startswith('negative') else [1, 0]
        return [SimpleNamespace(embedding=np.array(vector), bbox=np.array([0, 0, 20, 20]), det_score=.9)]


class JobsTests(unittest.TestCase):
    def fixture(self, base):
        data = {}
        for key, count in [('references', 2), ('positive', 3), ('negative', 3)]:
            data[key] = []
            for n in range(count):
                file = base / f'{key}{n}.jpg'
                file.write_bytes(file.name.encode())
                data[key].append(str(file))
        photos = base / 'photos'
        photos.mkdir()
        (photos / 'found.jpg').write_bytes(b'photo')
        return dict(data, photos=str(photos), output=str(base / 'results'), mode='scan',
                    config=dict(core.DEFAULTS, device='cpu', model_dir=str(base / 'models')))

    def test_minimum_and_duplicate_path(self):
        with tempfile.TemporaryDirectory() as directory:
            job = self.fixture(Path(directory))
            job['negative'] = job['negative'][:2]
            with self.assertRaises(ValueError):
                validate(job)
            job['negative'].append(job['references'][0])
            with self.assertRaises(ValueError):
                validate(job)

    def test_check_then_scan_and_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            job = self.fixture(Path(directory))
            events = []
            run_job(job, lambda *event: events.append(event), threading.Event(), FakeEngine)
            run = Path(next(value for kind, value in events if kind == 'output'))
            self.assertEqual(json.loads((run / 'summary.json').read_text())['me'], 1)
            self.assertTrue((run / 'sample_check.json').is_file())
            self.assertEqual((run / 'me' / 'found.jpg').read_bytes(), b'photo')
            self.assertEqual((Path(job['photos']) / 'found.jpg').read_bytes(), b'photo')

    def test_cancel_closes_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            job = self.fixture(Path(directory))
            events = []
            stop = threading.Event()
            def emit(kind, value=None):
                events.append((kind, value))
                if kind == 'scan_progress':
                    stop.set()
            (Path(job['photos']) / 'second.jpg').write_bytes(b'second')
            with self.assertRaises(core.Cancelled):
                run_job(job, emit, stop, FakeEngine)
            run = Path(next(value for kind, value in events if kind == 'output'))
            self.assertEqual(len(json.loads((run / 'report.json').read_text())), 1)
            self.assertEqual(json.loads((run / 'status.json').read_text())['status'], 'cancelled')

    def test_renamed_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            job = self.fixture(Path(directory))
            Path(job['positive'][0]).write_bytes(Path(job['references'][0]).read_bytes())
            with self.assertRaises(ValueError):
                run_job(job, lambda *args: None, threading.Event(), FakeEngine)

    def test_wrong_sample_warns_but_scans_collection(self):
        class WrongEngine(FakeEngine):
            def faces(self, path):
                return super().faces(Path('positive.jpg'))
        with tempfile.TemporaryDirectory() as directory:
            job = self.fixture(Path(directory))
            events = []
            run_job(job, lambda *event: events.append(event), threading.Event(), WrongEngine)
            self.assertIn('sample_warning', [kind for kind, value in events])
            self.assertIn('summary', [kind for kind, value in events])

    def test_missed_positive_warns_but_scans_collection(self):
        class MissEngine(FakeEngine):
            def faces(self, path):
                return super().faces(Path('negative.jpg') if Path(path).name == 'positive0.jpg' else path)
        with tempfile.TemporaryDirectory() as directory:
            job = self.fixture(Path(directory))
            events = []
            run_job(job, lambda *event: events.append(event), threading.Event(), MissEngine)
            sample = next(value for kind, value in events if kind == 'sample_result')
            self.assertEqual(sample['positive']['not_me'], 1)
            self.assertEqual(sample['errors'], 0)
            self.assertIn('summary', [kind for kind, value in events])


if __name__ == '__main__':
    unittest.main()
