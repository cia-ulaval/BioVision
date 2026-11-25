import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np

from pipeline.extraction.extraction import trouver_csv_du_lot, lire_videos_du_lot, dossier_frames_de_video
from pipeline.extraction.extraction import extraire_frames_video, extraire_frames_des_lots
from pipeline.extraction import config

def ecrire_video(chemin, secondes, fps=15):
    # Petite vidéo de test : 15 fps comme les caméras des colliers
    writer = cv2.VideoWriter(str(chemin), cv2.VideoWriter_fourcc(*"MJPG"), fps, (32, 32))
    for index in range(secondes * fps):
        writer.write(np.full((32, 32, 3), index % 256, dtype=np.uint8))
    writer.release()
    return chemin

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

class TestDossierFramesDeVideo(unittest.TestCase):

    def test_dossier_frames_de_video(self):
        chemin_video = Path("/videos/annotations/to_analyse2/C00217_2021_03_12_193000_winter_20.avi")
        self.assertEqual(dossier_frames_de_video(chemin_video, Path("frames")),
                         Path("frames") / "C00217" / "C00217_2021_03_12_193000_winter_20")

    def test_dossier_frames_de_video_avec_chemin_en_texte(self):
        chemin_video = "/videos/annotations/to_analyse2/C00217_2021_03_12_193000_winter_20.avi"
        self.assertEqual(dossier_frames_de_video(chemin_video, "frames"),
                         Path("frames") / "C00217" / "C00217_2021_03_12_193000_winter_20")

    def test_dossier_frames_de_video_par_defaut(self):
        chemin_video = Path("C00225_2021_06_01_120000_summer_33.avi")
        self.assertEqual(dossier_frames_de_video(chemin_video),
                         config.PATH_TO_FRAMES / "C00225" / "C00225_2021_06_01_120000_summer_33")

    def test_dossier_frames_de_video_ne_cree_pas_le_dossier(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        dossier = dossier_frames_de_video("C00217_2021_03_12_193000_winter_20.avi", temp_dir.name)

        self.assertFalse(dossier.exists())

class TestExtraireFramesVideo(unittest.TestCase):

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.dossier = Path(temp_dir.name)
        self.sortie = self.dossier / "frames" / "C00217" / "video"

    def creer_video(self, secondes, fps=15):
        return ecrire_video(self.dossier / "video.avi", secondes, fps)

    def test_extraire_frames_video(self):
        frames = extraire_frames_video(self.creer_video(3), self.sortie, frames_per_second=1)

        self.assertEqual([frame.name for frame in frames], ["f00000.jpg", "f00015.jpg", "f00030.jpg"])
        self.assertEqual(sorted(self.sortie.glob("*.jpg")), frames)

    def test_extraire_frames_video_avec_autre_cadence(self):
        # 3 frames par seconde sur une vidéo à 15 fps : une frame sur 5
        frames = extraire_frames_video(self.creer_video(1), self.sortie, frames_per_second=3)
        self.assertEqual([frame.name for frame in frames], ["f00000.jpg", "f00005.jpg", "f00010.jpg"])

    def test_extraire_frames_video_deja_extraite(self):
        # Reprise : une vidéo déjà extraite n'est pas extraite à nouveau
        self.sortie.mkdir(parents=True)
        (self.sortie / "f00000.jpg").write_text("test")

        frames = extraire_frames_video(self.creer_video(3), self.sortie, frames_per_second=1)

        self.assertEqual(frames, [self.sortie / "f00000.jpg"])

    def test_extraire_frames_video_illisible(self):
        chemin = self.dossier / "pas_une_video.avi"
        chemin.write_text("test")

        with self.assertWarns(UserWarning):
            frames = extraire_frames_video(chemin, self.sortie)

        self.assertEqual(frames, [])
        self.assertFalse(self.sortie.exists())

class TestExtraireFramesDesLots(unittest.TestCase):

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.annotations = Path(temp_dir.name) / "annotations"
        self.frames = Path(temp_dir.name) / "frames"

    def creer_lot(self, lot, videos_presentes, videos_absentes=()):
        # Les vidéos absentes sont listées dans le CSV, mais pas écrites sur le disque
        dossier = self.annotations / lot
        dossier.mkdir(parents=True)
        for video in videos_presentes:
            ecrire_video(dossier / video, secondes=2)
        lines = ["RootFolder,File,RelativePath"] + [f"{lot},{video}," for video in [*videos_presentes, *videos_absentes]]
        (dossier / "labels.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def extraire(self, lots):
        # Toujours dans le dossier temporaire, jamais dans PATH_TO_FRAMES
        with redirect_stdout(io.StringIO()):
            return extraire_frames_des_lots(lots, self.annotations, self.frames)

    def test_extraire_frames_des_lots(self):
        self.creer_lot("to_analyse2", ["C00217_a.avi"])
        self.creer_lot("to_analyse3", ["C00218_b.avi"])

        frames_par_video = self.extraire(["to_analyse2", "to_analyse3"])

        self.assertEqual(frames_par_video, {
            "C00217_a": [self.frames / "C00217" / "C00217_a" / "f00000.jpg", self.frames / "C00217" / "C00217_a" / "f00015.jpg"],
            "C00218_b": [self.frames / "C00218" / "C00218_b" / "f00000.jpg", self.frames / "C00218" / "C00218_b" / "f00015.jpg"],
        })

    def test_extraire_frames_des_lots_avec_video_absente(self):
        # Une vidéo manquante n'arrête pas l'extraction des autres
        self.creer_lot("to_analyse2", ["C00217_a.avi"], videos_absentes=["C00217_absente.avi"])

        with self.assertWarns(UserWarning):
            frames_par_video = self.extraire(["to_analyse2"])

        self.assertEqual(list(frames_par_video), ["C00217_a"])

    def test_extraire_frames_des_lots_avec_video_dans_deux_lots(self):
        self.creer_lot("to_analyse2", ["C00217_a.avi"])
        self.creer_lot("to_analyse3", ["C00217_a.avi"])

        with self.assertWarns(UserWarning):
            frames_par_video = self.extraire(["to_analyse2", "to_analyse3"])

        self.assertEqual(list(frames_par_video), ["C00217_a"])

    def test_extraire_frames_des_lots_avec_lot_inexistant(self):
        with self.assertRaises(FileNotFoundError):
            self.extraire(["to_analyse_2"])
