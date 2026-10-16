import csv
import tempfile
import unittest
from pathlib import Path

from pipeline.frame_io import get_frames_from_folder, export_frame_results_csv, read_frame_results_csv, FrameResult

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

HEADER = ["frame_path", "sharpness", "darkness", "saturation", "contrast", "brightness", "result"]

class TestExportFrameResultsCSV(unittest.TestCase):
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

    def make_result(self, frame, result=True):
        # Scores fixes : seul le chemin et la décision varient selon les tests
        return FrameResult(frame, darkness=1.5, saturation=12.5, sharpness=80.25, contrast=30.5, brightness=128.0, result=result)

    def test_export_frame_results_csv(self):
        frame_results = [self.make_result(self.frame1), self.make_result(self.frame2, result=False)]

        exported = export_frame_results_csv(frame_results, base_folder=self.frames_folder, csv_path=self.csv_path)

        # Toutes les frames sont écrites, retenues ou non, avec leurs scores
        self.assertEqual(self.read_csv(self.csv_path), [
            HEADER,
            ["frame1.jpg", "80.25", "1.5", "12.5", "30.5", "128.0", "True"],
            ["frame2.png", "80.25", "1.5", "12.5", "30.5", "128.0", "False"]
        ])

        # Vérifier que la fonction retourne les résultats exportés
        self.assertEqual(exported, frame_results)

    def test_export_frame_results_csv_with_subfolders(self):
        frame = self.make_video_frame("video_01", "frame_0001.jpg")

        export_frame_results_csv([self.make_result(frame)], base_folder=self.frames_folder, csv_path=self.csv_path)

        # Le chemin garde le sous-dossier de la vidéo, toujours avec des "/"
        self.assertEqual(self.read_csv(self.csv_path), [
            HEADER,
            ["video_01/frame_0001.jpg", "80.25", "1.5", "12.5", "30.5", "128.0", "True"]
        ])

    def test_export_frame_results_csv_with_same_name_in_two_videos(self):
        frame_a = self.make_video_frame("video_01", "frame_0001.jpg")
        frame_b = self.make_video_frame("video_02", "frame_0001.jpg")

        export_frame_results_csv([self.make_result(frame_a), self.make_result(frame_b)],
                                 base_folder=self.frames_folder, csv_path=self.csv_path)

        # Les deux frames sont gardées, chacune avec sa vidéo d'origine
        self.assertEqual([row[0] for row in self.read_csv(self.csv_path)], [
            "frame_path",
            "video_01/frame_0001.jpg",
            "video_02/frame_0001.jpg"
        ])

    def test_export_frame_results_csv_with_empty_list(self):
        exported = export_frame_results_csv([], base_folder=self.frames_folder, csv_path=self.csv_path)

        # Seulement l'en-tête
        self.assertEqual(self.read_csv(self.csv_path), [HEADER])
        self.assertEqual(exported, [])

    def test_export_frame_results_csv_overwrites_existing_csv(self):
        self.csv_path.write_text("ancien contenu\n")

        export_frame_results_csv([self.make_result(self.frame1)], base_folder=self.frames_folder, csv_path=self.csv_path)

        # L'ancien contenu est remplacé (mode "w")
        self.assertEqual(self.read_csv(self.csv_path), [
            HEADER,
            ["frame1.jpg", "80.25", "1.5", "12.5", "30.5", "128.0", "True"]
        ])

    def test_export_frame_results_csv_creates_parent_folder(self):
        csv_path = self.test_path / "results" / "output.csv"

        # Vérifier que le dossier parent n'existe pas avant l'exportation
        self.assertFalse(csv_path.parent.exists())

        export_frame_results_csv([self.make_result(self.frame1)], base_folder=self.frames_folder, csv_path=csv_path)

        self.assertEqual(self.read_csv(csv_path), [
            HEADER,
            ["frame1.jpg", "80.25", "1.5", "12.5", "30.5", "128.0", "True"]
        ])

    def test_export_frame_results_csv_with_str_paths(self):
        frame_results = [self.make_result(str(self.frame1)), self.make_result(str(self.frame2))]

        export_frame_results_csv(frame_results, base_folder=str(self.frames_folder), csv_path=str(self.csv_path))

        self.assertEqual([row[0] for row in self.read_csv(self.csv_path)], [
            "frame_path",
            "frame1.jpg",
            "frame2.png"
        ])

    def test_export_frame_results_csv_with_frame_outside_base_folder(self):
        outside_frame = self.test_path / "outside.jpg"
        outside_frame.write_text("test")

        # relative_to() lève une ValueError si la frame n'est pas dans base_folder
        with self.assertRaises(ValueError):
            export_frame_results_csv([self.make_result(outside_frame)],
                                     base_folder=self.frames_folder, csv_path=self.csv_path)

class TestReadFrameResultsCSV(unittest.TestCase):
    def setUp(self):
        # Créer un dossier temporaire pour les tests
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)
        self.frames_folder = self.test_path / "frames"
        self.csv_path = self.test_path / "output.csv"

        # Une frame retenue et une rejetée, dont une dans un sous-dossier de vidéo
        self.good = FrameResult(self.frames_folder / "video_01" / "frame_0001.jpg",
                                darkness=1.5, saturation=12.5, sharpness=80.25, contrast=30.5, brightness=128.0, result=True)
        self.bad = FrameResult(self.frames_folder / "frame2.png",
                               darkness=30.0, saturation=99.75, sharpness=5.125, contrast=8.25, brightness=165.5, result=False)

    def tearDown(self):
        # Nettoyer le dossier temporaire après les tests
        self.test_dir.cleanup()

    def export(self, frame_results):
        export_frame_results_csv(frame_results, base_folder=self.frames_folder, csv_path=self.csv_path)

    def test_read_frame_results_csv_round_trip(self):
        self.export([self.good, self.bad])

        # Relire le CSV redonne exactement les mêmes résultats, dans le même ordre
        frame_results = read_frame_results_csv(self.csv_path, self.frames_folder)
        self.assertEqual(frame_results, [self.good, self.bad])

    def test_read_frame_results_csv_with_valid_mode(self):
        self.export([self.good, self.bad])

        frame_results = read_frame_results_csv(self.csv_path, self.frames_folder, filter_mode="valid")
        self.assertEqual(frame_results, [self.good])

    def test_read_frame_results_csv_converts_types(self):
        self.export([self.bad])

        # Les chaînes du CSV sont reconverties : "False" doit redonner False, pas True
        frame_result = read_frame_results_csv(self.csv_path, self.frames_folder)[0]
        self.assertIs(frame_result.result, False)
        self.assertIsInstance(frame_result.darkness, float)
        self.assertIsInstance(frame_result.saturation, float)
        self.assertIsInstance(frame_result.sharpness, float)
        self.assertIsInstance(frame_result.contrast, float)
        self.assertIsInstance(frame_result.brightness, float)
        self.assertIsInstance(frame_result.path, Path)

    def test_read_frame_results_csv_with_str_paths(self):
        self.export([self.good])

        frame_results = read_frame_results_csv(str(self.csv_path), str(self.frames_folder))
        self.assertEqual(frame_results, [self.good])

    def test_read_frame_results_csv_with_only_header(self):
        self.export([])

        self.assertEqual(read_frame_results_csv(self.csv_path, self.frames_folder), [])

    def test_read_frame_results_csv_with_invalid_filter_mode(self):
        self.export([self.good])

        with self.assertRaises(ValueError):
            read_frame_results_csv(self.csv_path, self.frames_folder, filter_mode="vaild")

    def test_read_frame_results_csv_with_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            read_frame_results_csv(self.test_path / "missing.csv", self.frames_folder)

    def test_read_frame_results_csv_without_saturation_column(self):
        # Un CSV d'une ancienne version du filtre (sans la colonne saturation) est refusé clairement
        self.csv_path.write_text("frame_path,sharpness,exposure,result\nframe2.png,5.125,30.0,False\n", encoding="utf-8")

        with self.assertRaises(ValueError) as context:
            read_frame_results_csv(self.csv_path, self.frames_folder)
        self.assertIn("saturation", str(context.exception))

    def test_read_frame_results_csv_without_contrast_and_brightness_columns(self):
        # Un CSV de la version du filtre sans le voile est refusé clairement : il faut relancer le filtre
        self.csv_path.write_text("frame_path,sharpness,darkness,saturation,result\nframe2.png,5.125,30.0,99.75,False\n",
                                 encoding="utf-8")

        with self.assertRaises(ValueError) as context:
            read_frame_results_csv(self.csv_path, self.frames_folder)
        self.assertIn("contrast", str(context.exception))
        self.assertIn("brightness", str(context.exception))
