import tempfile
import unittest
from pathlib import Path

from pipeline.frame_io import get_frames_from_folder, export_filtered_frames

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

class TestExportFilteredFrames(unittest.TestCase):
    def setUp(self):
        # Créer un dossier temporaire pour les tests
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

        # Créer un sous-dossier pour les frames à exporter
        self.frames_dir = self.test_path / "frames"
        self.frames_dir.mkdir()

        # Créer des fichiers de test à exporter
        self.frame1 = self.frames_dir / "frame1.jpg"
        self.frame2 = self.frames_dir / "frame2.png"
        self.frame1.write_text("test")
        self.frame2.write_text("test")

    def tearDown(self):
        # Nettoyer le dossier temporaire après les tests
        self.test_dir.cleanup()

    def test_export_filtered_frames(self):
        output_folder = self.test_path / "output"
        frames_to_export = [self.frame1, self.frame2]

        exported_frames = export_filtered_frames(frames_to_export, output_folder)

        # Vérifier que les frames ont été copiées dans le dossier de sortie
        self.assertTrue((output_folder / "frame1.jpg").exists())
        self.assertTrue((output_folder / "frame2.png").exists())

        # Vérifier que la liste des frames exportées correspond aux chemins attendus
        expected_exported_frames = sorted([
            output_folder / "frame1.jpg",
            output_folder / "frame2.png"
        ])
        self.assertEqual(sorted(exported_frames), expected_exported_frames)

    def test_export_filtered_frames_creates_output_folder(self):
        output_folder = self.test_path / "new_output"
        frames_to_export = [self.frame1]

        # Vérifier que le dossier de sortie n'existe pas avant l'exportation
        self.assertFalse(output_folder.exists())

        exported_frames = export_filtered_frames(frames_to_export, output_folder)

        # Vérifier que le dossier de sortie a été créé
        self.assertTrue(output_folder.exists())
        self.assertTrue((output_folder / "frame1.jpg").exists())

    def test_export_filtered_frames_with_empty_list(self):
        output_folder = self.test_path / "output"
        frames_to_export = []

        exported_frames = export_filtered_frames(frames_to_export, output_folder)

        # Vérifier que le dossier de sortie a été créé même si la liste est vide
        self.assertTrue(output_folder.exists())
        self.assertEqual(exported_frames, [])

    def test_export_filtered_frames_with_existent_output_folder(self):
        output_folder = self.test_path / "output"
        output_folder.mkdir()
        frames_to_export = [self.frame1]

        exported_frames = export_filtered_frames(frames_to_export, output_folder)

        # Vérifier que les frames ont été copiées dans le dossier de sortie
        self.assertTrue((output_folder / "frame1.jpg").exists())
        self.assertEqual(len(exported_frames), 1)

    def test_export_filtered_frames_with_str_frames(self):
        output_folder = self.test_path / "output"
        frames_to_export = [str(self.frame1), str(self.frame2)]

        exported_frames = export_filtered_frames(frames_to_export, output_folder)

        # Vérifier que les frames ont été copiées dans le dossier de sortie
        self.assertTrue((output_folder / "frame1.jpg").exists())
        self.assertTrue((output_folder / "frame2.png").exists())

        # Vérifier que la liste des frames exportées correspond aux chemins attendus
        expected_exported_frames = [
            output_folder / "frame1.jpg",
            output_folder / "frame2.png"
        ]
        self.assertEqual(exported_frames, expected_exported_frames)

    def test_export_filtered_frames_with_str_output_folder(self):
        output_folder = self.test_path / "output"
        frames_to_export = [self.frame1, self.frame2]

        exported_frames = export_filtered_frames(frames_to_export, str(output_folder))

        # Vérifier que les frames ont été copiées dans le dossier de sortie
        self.assertTrue((output_folder / "frame1.jpg").exists())
        self.assertTrue((output_folder / "frame2.png").exists())

        # Vérifier que la liste des frames exportées correspond aux chemins attendus
        expected_exported_frames = [
            output_folder / "frame1.jpg",
            output_folder / "frame2.png"
        ]
        self.assertEqual(exported_frames, expected_exported_frames)