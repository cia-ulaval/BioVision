import csv
import tempfile
import unittest
from pathlib import Path

from pipeline.frame_io import get_frames_from_folder, export_frames_csv

class TestGetFramesFromFolder(unittest.TestCase):
    def setUp(self):
        # Créer un dossier temporaire pour les tests
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

        # Définir les fichiers attendus pour les tests
        self.expected_default_frames = sorted([
            self.test_path / "frame1.jpg",
            self.test_path / "frame2.png",
            self.test_path / "frame4.JPG",
            self.test_path / "frame6.PNG"
        ])

        # Créer des fichiers de test avec différentes extensions
        self.test_path.joinpath("frame1.jpg").write_text("test")
        self.test_path.joinpath("frame2.png").write_text("test")
        self.test_path.joinpath("frame3.txt").write_text("test")  # Fichier non image
        self.test_path.joinpath("frame4.JPG").write_text("test")  # Extension en majuscule
        self.test_path.joinpath("frame5.jpeg").write_text("test") # Extension non incluse dans le filtre
        self.test_path.joinpath("frame6.PNG").write_text("test")  # Extension en majuscule
        self.test_path.joinpath("frame7.gif").write_text("test")  # Extension non incluse dans le filtre
        self.test_path.joinpath("frame8.bmp").write_text("test")  # Extension non incluse dans le filtre

    def tearDown(self):
        # Nettoyer le dossier temporaire après les tests
        self.test_dir.cleanup()

    def test_when_folder_does_not_exist(self):
        non_existent_path = self.test_path / "non_existent_folder"
        self.assertRaises(ValueError, get_frames_from_folder, non_existent_path)

    def test_when_folder_is_not_a_directory(self):
        file_path = self.test_path / "frame1.jpg"
        self.assertRaises(ValueError, get_frames_from_folder, file_path)

    def test_get_frames_with_default_extensions(self):
        frames = get_frames_from_folder(self.test_path)
        self.assertEqual(frames, self.expected_default_frames)

    def test_get_frames_with_custom_extensions(self):
        frames = get_frames_from_folder(self.test_path, extensions=(".txt", ".gif"))
        expected_frames = sorted([
            self.test_path / "frame3.txt",
            self.test_path / "frame7.gif"
        ])
        self.assertEqual(frames, expected_frames)

    def test_get_frames_with_single_extension_string(self):
        frames = get_frames_from_folder(self.test_path, extensions=".jpg")
        expected_frames = sorted([
            self.test_path / "frame1.jpg",
            self.test_path / "frame4.JPG"
        ])
        self.assertEqual(frames, expected_frames)

    def test_get_frames_with_extensions_without_dot(self):
        frames = get_frames_from_folder(self.test_path, extensions=("jpg", "png"))
        self.assertEqual(frames, self.expected_default_frames)

    def test_get_frames_with_majuscule_extensions(self):
        frames = get_frames_from_folder(self.test_path, extensions=(".JPG", ".PNG"))
        self.assertEqual(frames, self.expected_default_frames)

    def test_get_frames_with_string_path(self):
        frames = get_frames_from_folder(str(self.test_path))
        self.assertEqual(frames, self.expected_default_frames)

    def test_get_frames_ignores_directories_with_valid_extension(self):
        subfolder = self.test_path / "subfolder.jpg"
        subfolder.mkdir()

        frames = get_frames_from_folder(self.test_path)

        self.assertNotIn(subfolder, frames)

    def test_get_frames_with_no_matching_extensions(self):
        frames = get_frames_from_folder(self.test_path, extensions=(".tiff", ".webp"))
        self.assertEqual(frames, [])

    def test_get_frames_with_subfolder(self):
        # Créer un sous-dossier pour les tests
        test_subfolder = self.test_path / "subfolder"
        test_subfolder.mkdir(exist_ok=True)
        test_subfolder.joinpath("frame3.jpg").write_text("test")

        frames = get_frames_from_folder(self.test_path)
        expected_frames = sorted([
            self.test_path / "frame1.jpg",
            self.test_path / "frame2.png",
            self.test_path / "frame4.JPG",
            self.test_path / "frame6.PNG",
            test_subfolder / "frame3.jpg"
        ])
        self.assertEqual(frames, expected_frames)

class TestExportFramesCSV(unittest.TestCase):
    def setUp(self):
        # Créer un dossier temporaire pour les tests
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

        # Créer un sous-dossier pour les frames
        self.frames_folder = self.test_path / "frames"
        self.frames_folder.mkdir()

        # Créer des fichiers de test
        self.frame1 = self.frames_folder / "frame1.jpg"
        self.frame2 = self.frames_folder / "frame2.png"
        self.frame1.write_text("test")
        self.frame2.write_text("test")

        # Chemin du CSV de sortie
        self.csv_path = self.test_path / "output.csv"

    def tearDown(self):
        # Nettoyer le dossier temporaire après les tests
        self.test_dir.cleanup()

    def read_csv(self, path):
        # Relire le CSV sous forme de liste de lignes
        with open(path, mode='r', newline='', encoding='utf-8') as csvfile:
            return list(csv.reader(csvfile))

    def make_video_frame(self, video, name):
        # Créer une frame dans un sous-dossier de vidéo
        video_folder = self.frames_folder / video
        video_folder.mkdir(exist_ok=True)
        frame = video_folder / name
        frame.write_text("test")
        return frame

    def test_export_frames_csv(self):
        frames = [self.frame1, self.frame2]

        exported_frames = export_frames_csv(frames, base_folder=self.frames_folder, csv_path=self.csv_path)

        self.assertEqual(self.read_csv(self.csv_path), [
            ["frame_path"],
            ["frame1.jpg"],
            ["frame2.png"]
        ])

        # Vérifier que la fonction retourne la liste des frames exportées
        self.assertEqual(exported_frames, frames)

    def test_export_frames_csv_with_subfolders(self):
        frame = self.make_video_frame("video_01", "frame_0001.jpg")

        export_frames_csv([frame], base_folder=self.frames_folder, csv_path=self.csv_path)

        # Le chemin garde le sous-dossier de la vidéo, toujours avec des "/"
        self.assertEqual(self.read_csv(self.csv_path), [
            ["frame_path"],
            ["video_01/frame_0001.jpg"]
        ])

    def test_export_frames_csv_with_same_name_in_two_videos(self):
        frame_a = self.make_video_frame("video_01", "frame_0001.jpg")
        frame_b = self.make_video_frame("video_02", "frame_0001.jpg")

        export_frames_csv([frame_a, frame_b], base_folder=self.frames_folder, csv_path=self.csv_path)

        # Les deux frames sont gardées, chacune avec sa vidéo d'origine
        self.assertEqual(self.read_csv(self.csv_path), [
            ["frame_path"],
            ["video_01/frame_0001.jpg"],
            ["video_02/frame_0001.jpg"]
        ])

    def test_export_frames_csv_with_empty_list(self):
        exported_frames = export_frames_csv([], base_folder=self.frames_folder, csv_path=self.csv_path)

        # Seulement l'en-tête
        self.assertEqual(self.read_csv(self.csv_path), [["frame_path"]])
        self.assertEqual(exported_frames, [])

    def test_export_frames_csv_overwrites_existing_csv(self):
        self.csv_path.write_text("ancien contenu\n")

        export_frames_csv([self.frame1], base_folder=self.frames_folder, csv_path=self.csv_path)

        # L'ancien contenu est remplacé (mode "w")
        self.assertEqual(self.read_csv(self.csv_path), [
            ["frame_path"],
            ["frame1.jpg"]
        ])

    def test_export_frames_csv_creates_parent_folder(self):
        csv_path = self.test_path / "results" / "output.csv"

        # Vérifier que le dossier parent n'existe pas avant l'exportation
        self.assertFalse(csv_path.parent.exists())

        export_frames_csv([self.frame1], base_folder=self.frames_folder, csv_path=csv_path)

        self.assertEqual(self.read_csv(csv_path), [
            ["frame_path"],
            ["frame1.jpg"]
        ])

    def test_export_frames_csv_with_str_paths(self):
        frames = [str(self.frame1), str(self.frame2)]

        export_frames_csv(frames, base_folder=str(self.frames_folder), csv_path=str(self.csv_path))

        self.assertEqual(self.read_csv(self.csv_path), [
            ["frame_path"],
            ["frame1.jpg"],
            ["frame2.png"]
        ])

    def test_export_frames_csv_with_frame_outside_base_folder(self):
        outside_frame = self.test_path / "outside.jpg"
        outside_frame.write_text("test")

        # relative_to() lève une ValueError si la frame n'est pas dans base_folder
        with self.assertRaises(ValueError):
            export_frames_csv([outside_frame], base_folder=self.frames_folder, csv_path=self.csv_path)