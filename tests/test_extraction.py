import tempfile
import unittest
from pathlib import Path

from pipeline.extraction.extraction import trouver_csv_du_lot, lire_videos_du_lot

class TestTrouverCsvDuLot(unittest.TestCase):

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.annotations = Path(temp_dir.name)

    def creer_lot(self, lot, fichiers):
        dossier = self.annotations / lot
        dossier.mkdir(parents=True)
        for fichier in fichiers:
            (dossier / fichier).parent.mkdir(parents=True, exist_ok=True)
            (dossier / fichier).write_text("test")

    def test_trouver_csv_du_lot(self):
        self.creer_lot("to_analyse2", ["labels.csv", "video.avi"])
        self.assertEqual(trouver_csv_du_lot("to_analyse2", self.annotations), self.annotations / "to_analyse2" / "labels.csv")

    def test_trouver_csv_du_lot_dans_un_sous_dossier(self):
        self.creer_lot("to_analyse/Caribou_rand_Fanny1", ["labels.csv"])
        self.assertEqual(trouver_csv_du_lot("to_analyse/Caribou_rand_Fanny1", self.annotations),
                         self.annotations / "to_analyse" / "Caribou_rand_Fanny1" / "labels.csv")

    def test_trouver_csv_du_lot_ignore_les_sauvegardes(self):
        # Les copies de Timelapse dans Backups/ ne comptent pas comme un deuxième CSV
        self.creer_lot("to_analyse2", ["labels.csv", "Backups/labels_ancien.csv"])
        self.assertEqual(trouver_csv_du_lot("to_analyse2", self.annotations), self.annotations / "to_analyse2" / "labels.csv")

    def test_trouver_csv_du_lot_sans_csv(self):
        self.creer_lot("to_analyse2", ["video.avi"])
        with self.assertRaises(FileNotFoundError):
            trouver_csv_du_lot("to_analyse2", self.annotations)

    def test_trouver_csv_du_lot_avec_plusieurs_csv(self):
        self.creer_lot("to_analyse2", ["labels_a.csv", "labels_b.csv"])
        with self.assertRaises(ValueError):
            trouver_csv_du_lot("to_analyse2", self.annotations)

    def test_trouver_csv_du_lot_inexistant(self):
        with self.assertRaises(FileNotFoundError):
            trouver_csv_du_lot("to_analyse_2", self.annotations)

class TestLireVideosDuLot(unittest.TestCase):

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.lot = Path(temp_dir.name)
        self.csv_path = self.lot / "labels.csv"

    def write_csv(self, rows):
        # Mêmes premières colonnes que le fichier des biologistes
        lines = ["RootFolder,File,RelativePath,NotUsable"]
        lines += [f"to_analyse2,{file},{relative_path},false" for file, relative_path in rows]
        self.csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def creer_video(self, chemin):
        (self.lot / chemin).parent.mkdir(parents=True, exist_ok=True)
        (self.lot / chemin).write_text("test")

    def test_lire_videos_du_lot(self):
        self.creer_video("video_a.avi")
        self.creer_video("video_b.avi")
        self.write_csv([("video_a.avi", ""), ("video_b.avi", "")])

        self.assertEqual(lire_videos_du_lot(self.csv_path),
                         {"video_a": self.lot / "video_a.avi", "video_b": self.lot / "video_b.avi"})

    def test_lire_videos_du_lot_avec_chemin_relatif_windows(self):
        # Timelapse écrit les sous-dossiers avec des \
        self.creer_video("dossier/sous_dossier/video_a.avi")
        self.write_csv([("video_a.avi", "dossier\\sous_dossier")])

        self.assertEqual(lire_videos_du_lot(self.csv_path),
                         {"video_a": self.lot / "dossier" / "sous_dossier" / "video_a.avi"})

    def test_lire_videos_du_lot_avec_video_introuvable(self):
        self.creer_video("video_a.avi")
        self.write_csv([("video_a.avi", ""), ("video_absente.avi", "")])

        with self.assertWarns(UserWarning):
            videos = lire_videos_du_lot(self.csv_path)

        self.assertEqual(videos, {"video_a": self.lot / "video_a.avi"})

    def test_lire_videos_du_lot_avec_video_en_double(self):
        self.creer_video("video_a.avi")
        self.write_csv([("video_a.avi", ""), ("video_a.avi", "")])

        with self.assertWarns(UserWarning):
            videos = lire_videos_du_lot(self.csv_path)

        self.assertEqual(videos, {"video_a": self.lot / "video_a.avi"})
