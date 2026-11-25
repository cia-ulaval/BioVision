import tempfile
import unittest
from pathlib import Path

from pipeline.extraction.extraction import trouver_csv_du_lot

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
