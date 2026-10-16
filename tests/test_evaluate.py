import tempfile
import unittest
from pathlib import Path

from pipeline.evaluate import read_not_usable_labels, video_rejection_percents, predict_not_usable, VIDEO_REJECT_PERCENT
from pipeline.frame_io import FrameResult

class TestReadNotUsableLabels(unittest.TestCase):

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.csv_path = Path(temp_dir.name) / "labels.csv"

    def write_csv(self, rows):
        # Mêmes premières colonnes que le fichier des biologistes
        lines = ["RootFolder,File,RelativePath,DateTime,NotUsable,Behaviour"]
        lines += [f"to_analyse2,{file},, 2026-02-24 13:38:42,{not_usable},{behaviour}" for file, not_usable, behaviour in rows]
        self.csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_read_not_usable_labels(self):
        self.write_csv([
            ("C00217_2022_03_12_160003_winter_212.avi", "true", ""),
            ("C00219_2021_03_13_223000_winter_212.avi", "false", "Feeding"),
        ])

        labels = read_not_usable_labels(self.csv_path)

        self.assertEqual(labels, {
            "C00217_2022_03_12_160003_winter_212": True,
            "C00219_2021_03_13_223000_winter_212": False,
        })

    def test_read_not_usable_labels_with_multiple_values_in_a_column(self):
        # Une valeur entre guillemets avec une virgule (ex. plusieurs comportements) ne décale pas les colonnes
        self.write_csv([("C00219_2021_03_13_223000_winter_212.avi", "false", '"Feeding,Moving (walking)"')])

        labels = read_not_usable_labels(self.csv_path)

        self.assertEqual(labels, {"C00219_2021_03_13_223000_winter_212": False})

    def test_read_not_usable_labels_with_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            read_not_usable_labels(self.csv_path)

class TestVideoRejectionPercents(unittest.TestCase):

    def make_frame_result(self, path, result):
        # Les scores ne servent pas au calcul : seul result compte
        return FrameResult(path=Path(path), darkness=0.0, saturation=0.0, sharpness=0.0, contrast=0.0, brightness=0.0,
                           result=result)

    def test_video_rejection_percents(self):
        frame_results = [
            self.make_frame_result("frames/video_a/f00000.jpg", True),
            self.make_frame_result("frames/video_a/f00001.jpg", False),
            self.make_frame_result("frames/video_a/f00002.jpg", False),
            self.make_frame_result("frames/video_a/f00003.jpg", False),
            self.make_frame_result("frames/video_b/f00000.jpg", True),
        ]

        self.assertEqual(video_rejection_percents(frame_results), {"video_a": 75.0, "video_b": 0.0})

    def test_video_rejection_percents_with_all_frames_rejected(self):
        frame_results = [self.make_frame_result("frames/video_a/f00000.jpg", False)]
        self.assertEqual(video_rejection_percents(frame_results), {"video_a": 100.0})

    def test_video_rejection_percents_with_no_frames(self):
        self.assertEqual(video_rejection_percents([]), {})

class TestPredictNotUsable(unittest.TestCase):

    def test_predict_not_usable(self):
        rejection_percents = {"video_a": 0.0, "video_b": 100.0}
        self.assertEqual(predict_not_usable(rejection_percents), {"video_a": False, "video_b": True})

    def test_predict_not_usable_at_threshold(self):
        # Exactement au seuil : utilisable (on rejette strictement au-dessus)
        rejection_percents = {"at": VIDEO_REJECT_PERCENT, "above": VIDEO_REJECT_PERCENT + 1}
        self.assertEqual(predict_not_usable(rejection_percents), {"at": False, "above": True})
