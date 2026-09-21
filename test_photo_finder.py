"""Unit/integration tests with synthetic embeddings; no model downloads required."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import photo_finder as pf


def face(vector):
    return SimpleNamespace(embedding=np.array(vector, dtype=np.float32),
                           bbox=np.array([1, 2, 30, 40]), det_score=.9)


class Tests(unittest.TestCase):
    def test_threshold_boundaries(self):
        self.assertEqual(pf.classify(.5, .38, .5), 'me')
        self.assertEqual(pf.classify(.38, .38, .5), 'uncertain')
        self.assertEqual(pf.classify(.379, .38, .5), 'not_me')
        self.assertEqual(pf.classify(None, .38, .5), 'uncertain')

    def test_all_faces_all_references(self):
        engine = SimpleNamespace(faces=lambda _: [face([1, 0]), face([0, 1])])
        refs = np.array([[-1, 0], [0, 1]], dtype=np.float32)
        row = pf.analyze(Path('x.jpg'), Path('.'), engine, refs, ['a', 'b'], pf.DEFAULTS)
        self.assertTrue(row['me_detected'])
        self.assertEqual(row['faces_detected'], 2)
        self.assertEqual(row['faces'][1]['reference'], 'b')
        self.assertEqual(row['best_similarity'], 1)

    def test_invalid_embeddings(self):
        for value in ([0, 0], [float('nan'), 1], [[1, 2]]):
            with self.assertRaises(ValueError):
                pf.normalize(value)

    def test_copy_nested_unicode_errors_and_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source, run = base / 'фото', base / 'result'
            run.mkdir()
            paths = ['a/одинаково.JPG', 'b/одинаково.JPG', 'empty.heic', 'broken.png']
            for value in paths:
                target = source / value
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(value.encode())
            original = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
            def infer(path):
                if path.name == 'broken.png':
                    raise OSError('synthetic decode failure')
                return [] if path.name == 'empty.heic' else [face([1, 0])]
            rows, counts = pf.scan(source, SimpleNamespace(faces=infer), np.array([[1, 0]]),
                                   ['reference'], pf.DEFAULTS, run)
            self.assertEqual(counts['me'], 2)
            self.assertEqual(counts['uncertain'], 2)
            self.assertEqual(counts['errors'], 1)
            self.assertEqual(len(json.loads((run / 'report.json').read_text(encoding='utf-8'))), 4)
            for row in rows:
                self.assertEqual(Path(row['output_path']).read_bytes(), (source / row['relative_path']).read_bytes())
            self.assertEqual(original, {p: p.read_bytes() for p in original})

    def test_reference_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            refs, run = base / 'refs', base / 'out'
            refs.mkdir()
            run.mkdir()
            for name in ['solo.jpg', 'group.jpg', 'none.jpg']:
                (refs / name).touch()
            engine = SimpleNamespace(faces=lambda p: {'solo.jpg': [face([1, 0])],
                       'group.jpg': [face([1, 0]), face([0, 1])], 'none.jpg': []}[p.name])
            matrix, names = pf.references(engine, refs, run)
            self.assertEqual(matrix.shape, (1, 2))
            self.assertEqual(len(names), 1)

    def test_exif_and_bgr(self):
        # Only the optional plugin registration is mocked; Pillow actually decodes PNG.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ориентация.png'
            image = Image.new('RGB', (3, 2), (255, 0, 0))
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, exif=exif)
            with patch.dict(sys.modules, {'pillow_heif': SimpleNamespace(register_heif_opener=lambda: None)}):
                result = pf.load_image(path)
            self.assertEqual(result.shape, (3, 2, 3))
            self.assertEqual(result[0, 0].tolist(), [0, 0, 255])
            self.assertTrue(result.flags.c_contiguous)

    def test_calibration_no_face_in_denominator(self):
        positive = [dict(best_similarity=.8, processing_status='ok'),
                    dict(best_similarity=None, processing_status='no_faces')]
        negative = [dict(best_similarity=.1, processing_status='ok')]
        metric = pf.calibration_metrics(positive, negative, .38, .5)
        self.assertEqual(metric['retained_positive_recall'], 1)
        self.assertEqual(metric['confident_positive_recall'], .5)
        self.assertEqual(pf.distribution(positive)['no_faces'], 1)


if __name__ == '__main__':
    unittest.main()
