import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from pipeline.filtrage.evaluate import read_not_usable_labels, video_rejection_percents, predict_not_usable
from pipeline.filtrage.evaluate import calculate_recall, calculate_precision, find_compared_videos
from pipeline.filtrage.evaluate import build_video_reports, export_video_csv, evaluate_filter
from pipeline.filtrage.frame_io import export_frame_results_csv
from pipeline.filtrage import config
from pipeline.filtrage.models import FrameResult, Cause

# Exemple commun aux tests du rappel, de la précision et des vidéos comparées :
# a, b, c : rejetées par le filtre et NotUsable (vrais positifs)
# d : rejetée par le filtre, mais utilisable (faux positif)
# e : gardée par le filtre, mais NotUsable (faux négatif)
# f : gardée et utilisable (vrai négatif)
# g : rejetée par le filtre, sans étiquette ; h : NotUsable, sans prédiction
PREDICTIONS = {"a": True, "b": True, "c": True, "d": True, "e": False, "f": False, "g": True}
LABELS = {"a": True, "b": True, "c": True, "d": False, "e": True, "f": False, "h": True}

def make_frame(video, index, cause, frames_folder=Path("frames")):
    # Les scores ne servent pas ici : seuls la vidéo et la cause comptent
    return FrameResult(path=Path(frames_folder) / video / f"f{index:05d}.jpg", darkness=0.0, saturation=0.0,
                       sharpness=0.0, contrast=0.0, brightness=0.0, result=cause is None, cause=cause)

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
        # Les scores et la cause ne servent pas au calcul : seul result compte
        return FrameResult(path=Path(path), darkness=0.0, saturation=0.0, sharpness=0.0, contrast=0.0, brightness=0.0,
                           result=result, cause=None if result else Cause.VOILEE)

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
        rejection_percents = {"at": config.VIDEO_REJECT_PERCENT, "above": config.VIDEO_REJECT_PERCENT + 1}
        self.assertEqual(predict_not_usable(rejection_percents), {"at": False, "above": True})

class TestCalculateRecall(unittest.TestCase):

    def test_calculate_recall(self):
        # 3 vrais positifs (a, b, c) et 1 faux négatif (e). h n'a pas de prédiction : ignorée
        self.assertEqual(calculate_recall(PREDICTIONS, LABELS), 0.75)

    def test_calculate_recall_without_not_usable_video(self):
        # Aucune vidéo NotUsable : le rappel n'a pas de sens
        self.assertIsNone(calculate_recall({"a": True}, {"a": False}))

class TestCalculatePrecision(unittest.TestCase):

    def test_calculate_precision(self):
        # 3 vrais positifs (a, b, c) et 1 faux positif (d). g n'a pas d'étiquette : ignorée, pas un faux positif
        self.assertEqual(calculate_precision(PREDICTIONS, LABELS), 0.75)

    def test_calculate_precision_without_rejected_video(self):
        # Le filtre ne rejette aucune vidéo étiquetée : la précision n'a pas de sens
        self.assertIsNone(calculate_precision({"a": False, "g": True}, {"a": True}))

class TestFindComparedVideos(unittest.TestCase):

    def test_find_compared_videos(self):
        # g (sans étiquette) et h (sans prédiction) ne sont pas comparées
        self.assertEqual(find_compared_videos(PREDICTIONS, LABELS), {"a", "b", "c", "d", "e", "f"})

class TestBuildVideoReports(unittest.TestCase):

    def test_build_video_reports(self):
        # 10 frames : 3 brûlées, 1 voilée, 6 retenues
        causes = [Cause.BRULEE] * 3 + [Cause.VOILEE] + [None] * 6
        frame_results = [make_frame("video_a", index, cause) for index, cause in enumerate(causes)]

        [report] = build_video_reports(frame_results)

        self.assertEqual(report.video_name, "video_a")
        self.assertEqual(report.frame_count, 10)
        self.assertEqual(report.rejected_percent, 40.0)
        # Une entrée par cause, même à 0 %, et leur somme est égale au pourcentage rejeté
        self.assertEqual(report.cause_percents,
                         {Cause.BRULEE: 30.0, Cause.NOIRE: 0.0, Cause.VOILEE: 10.0, Cause.VIDEO_COURTE: 0.0})
        self.assertEqual(sum(report.cause_percents.values()), report.rejected_percent)

    def test_build_video_reports_with_several_videos(self):
        frame_results = [make_frame("video_a", 0, None), make_frame("video_b", 0, Cause.NOIRE)]

        reports = build_video_reports(frame_results)

        self.assertEqual([report.video_name for report in reports], ["video_a", "video_b"])
        self.assertEqual([report.rejected_percent for report in reports], [0.0, 100.0])

class TestExportVideoCsv(unittest.TestCase):

    def test_export_video_csv(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        csv_path = Path(temp_dir.name) / "results" / "rapport.csv"
        reports = build_video_reports([make_frame("video_a", 0, Cause.BRULEE), make_frame("video_b", 0, None)])

        export_video_csv(reports, {"video_a": True, "video_b": False}, {"video_a": True}, csv_path)

        with open(csv_path, newline="", encoding="utf-8") as csvfile:
            rows = list(csv.reader(csvfile))
        self.assertEqual(rows[0], ["video", "nb_frames", "pct_rejetees", "pct_brulee", "pct_noire", "pct_voilee",
                                   "pct_video_courte", "prediction_not_usable", "label_biologistes"])
        self.assertEqual(rows[1], ["video_a", "1", "100.0", "100.0", "0.0", "0.0", "0.0", "True", "True"])
        # video_b n'a pas d'étiquette : case vide
        self.assertEqual(rows[2], ["video_b", "1", "0.0", "0.0", "0.0", "0.0", "0.0", "False", ""])

class TestEvaluateFilter(unittest.TestCase):

    def test_evaluate_filter(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        folder = Path(temp_dir.name)
        frames_folder = folder / "frames"
        frames_csv = folder / "frames.csv"
        labels_csv = folder / "labels.csv"
        report_csv = folder / "rapport.csv"

        # video_a : rejetée par le filtre et NotUsable ; video_b : gardée, mais NotUsable
        frame_results = [make_frame("video_a", 0, Cause.BRULEE, frames_folder), make_frame("video_b", 0, None, frames_folder)]
        export_frame_results_csv(frame_results, frames_folder, frames_csv)
        labels_csv.write_text("File,NotUsable\nvideo_a.avi,true\nvideo_b.avi,true\n", encoding="utf-8")

        with redirect_stdout(io.StringIO()):
            results = evaluate_filter(frames_csv, frames_folder, labels_csv, report_csv)

        self.assertEqual(results, {"recall": 0.5, "precision": 1.0, "compared_videos": {"video_a", "video_b"}})
        self.assertTrue(report_csv.exists())
