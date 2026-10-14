import csv
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
import cv2

from pipeline.filter_frame import prepare_frame, validate_frame_quality, calculate_darkness_score, calculate_sharpness_score, calculate_saturation_score, filter_frames
from pipeline.filter_frame import DARK_PIXEL_THRESHOLD, DARK_PERCENT_THRESHOLD, SATURATED_PIXEL_THRESHOLD, SATURATED_PERCENT_THRESHOLD
from pipeline.frame_io import FrameResult

class TestPrepareFrame(unittest.TestCase):

    def test_prepare_frame_with_color_image(self):
        # Une image couleur de 100x200 pixels
        img = np.full((100, 200, 3), 128, dtype=np.uint8)
    
        # Créer un fichier temporaire pour l'image
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "color_image.png")
            cv2.imwrite(path, img)
    
            gray_img = prepare_frame(path)
    
            self.assertIsNotNone(gray_img)
            self.assertEqual(gray_img.shape, (100, 200))
            self.assertTrue(np.array_equal(gray_img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

    def test_prepare_frame_with_already_grayscale_image(self):
        # Une image gris moyen de 100x200 pixels
        img = np.full((100, 200), 128, dtype=np.uint8)

        # Créer un fichier temporaire pour l'image
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "test_image.png")
            cv2.imwrite(path, img)

            gray_img = prepare_frame(path)

            self.assertIsNotNone(gray_img)
            self.assertEqual(gray_img.shape, (100, 200))
            self.assertTrue(np.array_equal(gray_img, img))

    def test_prepare_frame_with_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            prepare_frame("nonexistent_file.png")

    def test_prepare_frame_with_invalid_image(self):
        # Créer un fichier temporaire vide
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "invalid_image.png")
            with open(path, 'w') as f:
                f.write("This is not a valid image file.")

            with self.assertRaises(FileNotFoundError):
                prepare_frame(path)

    def test_prepare_frame_with_large_portrait_image(self):
        # Une image de 2000x1000 pixels
        img = np.full((2000, 1000), 128, dtype=np.uint8)

        # Créer un fichier temporaire pour l'image
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "large_image.png")
            cv2.imwrite(path, img)

            gray_img = prepare_frame(path)

            self.assertIsNotNone(gray_img)
            self.assertEqual(gray_img.shape, (640, 320))

    def test_prepare_frame_with_small_image(self):
        # Une image de 50x50 pixels
        img = np.full((50, 50), 128, dtype=np.uint8)

        # Créer un fichier temporaire pour l'image
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "small_image.png")
            cv2.imwrite(path, img)

            gray_img = prepare_frame(path)

            self.assertIsNotNone(gray_img)
            self.assertEqual(gray_img.shape, (50, 50))

    def test_prepare_frame_with_640x640_image(self):
        # Une image de 640x640 pixels
        img = np.full((640, 640), 128, dtype=np.uint8)

        # Créer un fichier temporaire pour l'image
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "exact_image.png")
            cv2.imwrite(path, img)

            gray_img = prepare_frame(path)

            self.assertIsNotNone(gray_img)
            self.assertEqual(gray_img.shape, (640, 640))

    def test_prepare_frame_with_large_landscape_image(self):
        # Une image de 1000x2000 pixels
        img = np.full((1000, 2000), 128, dtype=np.uint8)

        # Créer un fichier temporaire pour l'image
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "large_image.png")
            cv2.imwrite(path, img)

            gray_img = prepare_frame(path)

            self.assertIsNotNone(gray_img)
            self.assertEqual(gray_img.shape, (320, 640))

class TestCalculateSharpnessScore(unittest.TestCase):

    def make_checkerboard(self):
        # Créer une image nette (un motif de damier noir et blanc)
        img = np.zeros((100, 100), dtype=np.uint8)
        img[::2, ::2] = 255
        img[1::2, 1::2] = 255
        return img

    def test_sharpness_score_with_sharp_image(self):
        # Une image nette doit avoir un score de netteté positif
        sharp_img = self.make_checkerboard()
        score = calculate_sharpness_score(sharp_img)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0)

    def test_sharpness_score_blurry_lower_than_sharp(self):
        # La même image floutée doit avoir un score plus bas que l'originale
        sharp_img = self.make_checkerboard()
        blurry_img = cv2.GaussianBlur(sharp_img, (5, 5), 0)
        self.assertLess(calculate_sharpness_score(blurry_img), calculate_sharpness_score(sharp_img))

    def test_sharpness_score_with_uniform_image(self):
        # Une image uniforme (tous les pixels ont la même valeur) doit avoir un score de netteté de 0
        uniform_img = np.full((100, 100), 128, dtype=np.uint8)
        self.assertEqual(calculate_sharpness_score(uniform_img), 0.0)

    def test_sharpness_score_with_color_image(self):
        # Tester que la fonction lève une exception pour une image couleur
        color_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            calculate_sharpness_score(color_img)

    def test_sharpness_score_with_none_image(self):
        # Tester que la fonction lève une exception pour une image None
        with self.assertRaises(ValueError):
            calculate_sharpness_score(None)

class TestCalculateDarknessScore(unittest.TestCase):

    def test_darkness_score_with_black_image(self):
        # Une image noire a tous ses pixels noirs
        black_img = np.zeros((100, 100), dtype=np.uint8)
        score = calculate_darkness_score(black_img)
        self.assertIsInstance(score, float)
        self.assertEqual(score, 100.0)

    def test_darkness_score_with_white_image(self):
        # Une image blanche n'a aucun pixel noir
        white_img = np.full((100, 100), 255, dtype=np.uint8)
        self.assertEqual(calculate_darkness_score(white_img), 0.0)

    def test_darkness_score_with_half_black_half_white_image(self):
        # Moitié noire, moitié blanche : 50 % de pixels noirs
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 255
        self.assertEqual(calculate_darkness_score(img), 50.0)

    def test_darkness_score_at_pixel_threshold(self):
        # Un pixel au seuil est noir (<=), un pixel juste au-dessus ne l'est pas
        at_threshold = np.full((100, 100), DARK_PIXEL_THRESHOLD, dtype=np.uint8)
        above_threshold = np.full((100, 100), DARK_PIXEL_THRESHOLD + 1, dtype=np.uint8)
        self.assertEqual(calculate_darkness_score(at_threshold), 100.0)
        self.assertEqual(calculate_darkness_score(above_threshold), 0.0)

    def test_darkness_score_with_dark_image_without_black_pixels(self):
        # Une image sombre (moyenne 45, comme une forêt dense) mais sans pixel noir a un score de 0
        img = np.full((100, 100), 15, dtype=np.uint8)
        img[:, 50:] = 75
        self.assertEqual(img.mean(), 45.0)
        self.assertEqual(calculate_darkness_score(img), 0.0)

    def test_darkness_score_with_color_image(self):
        # Tester que la fonction lève une exception pour une image couleur
        color_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            calculate_darkness_score(color_img)

    def test_darkness_score_with_none_image(self):
        # Tester que la fonction lève une exception pour une image None
        with self.assertRaises(ValueError):
            calculate_darkness_score(None)

class TestCalculateSaturationScore(unittest.TestCase):

    def test_saturation_score_with_black_image(self):
        # Une image noire n'a aucun pixel saturé
        black_img = np.zeros((100, 100), dtype=np.uint8)
        score = calculate_saturation_score(black_img)
        self.assertIsInstance(score, float)
        self.assertEqual(score, 0.0)

    def test_saturation_score_with_white_image(self):
        # Une image blanche a tous ses pixels saturés
        white_img = np.full((100, 100), 255, dtype=np.uint8)
        self.assertEqual(calculate_saturation_score(white_img), 100.0)

    def test_saturation_score_with_half_black_half_white_image(self):
        # Moitié noire, moitié blanche : 50 % de pixels saturés
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 255
        self.assertEqual(calculate_saturation_score(img), 50.0)

    def test_saturation_score_at_pixel_threshold(self):
        # Un pixel au seuil est saturé (>=), un pixel juste en dessous ne l'est pas
        at_threshold = np.full((100, 100), SATURATED_PIXEL_THRESHOLD, dtype=np.uint8)
        below_threshold = np.full((100, 100), SATURATED_PIXEL_THRESHOLD - 1, dtype=np.uint8)
        self.assertEqual(calculate_saturation_score(at_threshold), 100.0)
        self.assertEqual(calculate_saturation_score(below_threshold), 0.0)

    def test_saturation_score_with_bright_image_without_saturated_pixels(self):
        # Une image claire (moyenne 200, comme la neige) mais sans pixel saturé a un score de 0
        img = np.full((100, 100), 170, dtype=np.uint8)
        img[:, 50:] = 230
        self.assertEqual(img.mean(), 200.0)
        self.assertEqual(calculate_saturation_score(img), 0.0)

    def test_saturation_score_with_color_image(self):
        # Tester que la fonction lève une exception pour une image couleur
        color_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            calculate_saturation_score(color_img)

    def test_saturation_score_with_none_image(self):
        # Tester que la fonction lève une exception pour une image None
        with self.assertRaises(ValueError):
            calculate_saturation_score(None)

class TestValidateFrameQuality(unittest.TestCase):

    def setUp(self):
        # Un dossier temporaire par test, supprimé automatiquement à la fin
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.temp_dir = temp_dir.name

    def write_image(self, img, name="frame.png"):
        # validate_frame_quality prend un chemin : on écrit l'image sur le disque
        path = os.path.join(self.temp_dir, name)
        cv2.imwrite(path, img)
        return path

    def make_checkerboard(self, low, high):
        # Damier net dont la luminosité moyenne vaut (low + high) / 2
        img = np.full((100, 100), low, dtype=np.uint8)
        img[::2, ::2] = high
        img[1::2, 1::2] = high
        return img

    def make_flat_with_filled_rows(self, percent, base, value):
        # Aplat sans texture valant `base`, dont les `percent` premières lignes sur 100 valent `value`.
        # base et value ne diffèrent que de 1 : la frontière ne crée presque pas de texture
        img = np.full((100, 100), base, dtype=np.uint8)
        img[:percent, :] = value
        return img

    def make_flat_dark(self, percent):
        # Aplat dont exactement `percent` % des pixels sont noirs
        return self.make_flat_with_filled_rows(percent, DARK_PIXEL_THRESHOLD + 1, DARK_PIXEL_THRESHOLD)

    def make_flat_saturated(self, percent):
        # Aplat dont exactement `percent` % des pixels sont saturés
        return self.make_flat_with_filled_rows(percent, SATURATED_PIXEL_THRESHOLD - 1, SATURATED_PIXEL_THRESHOLD)

    def test_validate_frame_quality_with_good_image(self):
        # Nette (damier), sans pixel noir ni saturé
        path = self.write_image(self.make_checkerboard(50, 200))
        self.assertTrue(validate_frame_quality(path).result)

    def test_validate_frame_quality_with_blurry_image(self):
        # Bien exposée (gris 128) mais floue (aucun contour, netteté 0) : gardée, le flou seul ne rejette jamais
        path = self.write_image(np.full((100, 100), 128, dtype=np.uint8))
        self.assertTrue(validate_frame_quality(path).result)

    def test_validate_frame_quality_returns_frame_result(self):
        # Un FrameResult avec tous les scores est retourné
        path = self.write_image(np.full((100, 100), 128, dtype=np.uint8))
        frame_result = validate_frame_quality(path)
        self.assertIsInstance(frame_result, FrameResult)
        self.assertEqual(frame_result.path, path)
        self.assertEqual(frame_result.darkness, 0.0)
        self.assertEqual(frame_result.saturation, 0.0)
        self.assertEqual(frame_result.sharpness, 0.0)
        self.assertTrue(frame_result.result)

    def test_validate_frame_quality_with_underexposed_image(self):
        # Sans texture et sous-exposée (10 points de pourcentage de pixels noirs au-dessus du seuil)
        path = self.write_image(self.make_flat_dark(int(DARK_PERCENT_THRESHOLD) + 10))
        self.assertFalse(validate_frame_quality(path).result)

    def test_validate_frame_quality_at_darkness_threshold(self):
        # Exactement au seuil : gardée (on rejette strictement au-dessus)
        path = self.write_image(self.make_flat_dark(int(DARK_PERCENT_THRESHOLD)))
        self.assertTrue(validate_frame_quality(path).result)

    def test_validate_frame_quality_just_above_darkness_threshold(self):
        # 1 point de pourcentage au-dessus du seuil : rejetée
        path = self.write_image(self.make_flat_dark(int(DARK_PERCENT_THRESHOLD) + 1))
        self.assertFalse(validate_frame_quality(path).result)

    def test_validate_frame_quality_with_black_image(self):
        # Entièrement noire et sans texture : rejetée
        path = self.write_image(np.full((100, 100), 5, dtype=np.uint8))
        self.assertFalse(validate_frame_quality(path).result)

    def test_validate_frame_quality_with_dark_textured_image(self):
        # 50 % de pixels noirs, mais texturée : gardée
        path = self.write_image(self.make_checkerboard(0, 40))
        self.assertTrue(validate_frame_quality(path).result)

    def test_validate_frame_quality_with_dark_forest_like_image(self):
        # Scène sombre comme une forêt dense : moyenne 45 (rejetée par l'ancien seuil de moyenne à 55),
        # mais nette et sans pixel noir, donc gardée
        img = self.make_checkerboard(15, 75)
        self.assertEqual(img.mean(), 45.0)
        path = self.write_image(img)
        self.assertTrue(validate_frame_quality(path).result)

    def test_validate_frame_quality_with_overexposed_image(self):
        # Sans texture et surexposée (10 points de pourcentage de pixels saturés au-dessus du seuil)
        path = self.write_image(self.make_flat_saturated(int(SATURATED_PERCENT_THRESHOLD) + 10))
        self.assertFalse(validate_frame_quality(path).result)

    def test_validate_frame_quality_at_saturation_threshold(self):
        # Exactement au seuil : gardée (on rejette strictement au-dessus)
        path = self.write_image(self.make_flat_saturated(int(SATURATED_PERCENT_THRESHOLD)))
        self.assertTrue(validate_frame_quality(path).result)

    def test_validate_frame_quality_just_above_saturation_threshold(self):
        # 1 point de pourcentage au-dessus du seuil : rejetée
        path = self.write_image(self.make_flat_saturated(int(SATURATED_PERCENT_THRESHOLD) + 1))
        self.assertFalse(validate_frame_quality(path).result)

    def test_validate_frame_quality_with_burned_image(self):
        # Entièrement brûlée et sans texture : rejetée
        path = self.write_image(np.full((100, 100), 250, dtype=np.uint8))
        self.assertFalse(validate_frame_quality(path).result)

    def test_validate_frame_quality_with_saturated_textured_image(self):
        # Neige au soleil : 100 % de pixels saturés, mais texturée, donc gardée
        path = self.write_image(self.make_checkerboard(240, 255))
        self.assertTrue(validate_frame_quality(path).result)

    def test_validate_frame_quality_with_bright_snow_like_image(self):
        # Scène claire comme la neige : moyenne 200 (rejetée par l'ancien seuil de moyenne à 160),
        # mais nette et sans pixel saturé, donc gardée
        img = self.make_checkerboard(170, 230)
        self.assertEqual(img.mean(), 200.0)
        path = self.write_image(img)
        self.assertTrue(validate_frame_quality(path).result)

    def test_validate_frame_quality_with_color_image(self):
        # Une image couleur est convertie en niveaux de gris : c'est une entrée valide
        color_img = cv2.cvtColor(self.make_checkerboard(50, 200), cv2.COLOR_GRAY2BGR)
        path = self.write_image(color_img)
        self.assertTrue(validate_frame_quality(path).result)

    def test_validate_frame_quality_with_unreadable_file(self):
        # Un fichier qui n'est pas une image fait lever l'exception de prepare_frame
        path = os.path.join(self.temp_dir, "invalid_image.png")
        with open(path, "w") as f:
            f.write("This is not a valid image file.")
        with self.assertRaises(FileNotFoundError):
            validate_frame_quality(path)

    def test_validate_frame_quality_with_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            validate_frame_quality(os.path.join(self.temp_dir, "missing.png"))

class TestFilterFrames(unittest.TestCase):

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.input_folder = Path(temp_dir.name) / "frames"
        self.input_folder.mkdir()
        self.output_folder = Path(temp_dir.name) / "output"

    def write_frame(self, relative_path, textured=True, saturated_percent=0):
        # Damier texturé (toujours gardé) ou aplat sans texture juste sous le seuil de saturation.
        # saturated_percent : nombre de lignes saturées sur 100, donc pourcentage de pixels saturés
        if textured:
            img = np.full((100, 100), 50, dtype=np.uint8)
            img[::2, ::2] = 200
            img[1::2, 1::2] = 200
        else:
            img = np.full((100, 100), SATURATED_PIXEL_THRESHOLD - 1, dtype=np.uint8)
        img[:saturated_percent, :] = SATURATED_PIXEL_THRESHOLD
        path = self.input_folder / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)
        return path

    def run_filter_frames(self, csv_path=None):
        # Retourne le résultat et les messages affichés
        output = io.StringIO()
        with redirect_stdout(output):
            frames = filter_frames(self.input_folder, csv_path)
        return frames, output.getvalue()

    def read_csv(self, path):
        with open(path, mode='r', newline='', encoding='utf-8') as csvfile:
            return list(csv.reader(csvfile))

    def test_filter_frames_without_csv_path(self):
        good = self.write_frame("good.png")

        frames, output = self.run_filter_frames()

        self.assertEqual(frames, [good])
        self.assertIn("Aucun CSV exporté", output)
        self.assertFalse(self.output_folder.exists())

    def test_filter_frames_exports_csv(self):
        good = self.write_frame("good.png")
        csv_path = self.output_folder / "frames.csv"

        frames, output = self.run_filter_frames(csv_path)

        self.assertEqual(frames, [good])
        self.assertIn("CSV exporté", output)
        self.assertEqual([row[0] for row in self.read_csv(csv_path)], ["frame_path", "good.png"])

    def test_filter_frames_when_export_fails(self):
        good = self.write_frame("good.png")
        self.output_folder.mkdir()

        # csv_path est un dossier : l'écriture échoue
        frames, output = self.run_filter_frames(self.output_folder)

        self.assertEqual(frames, [good])
        self.assertIn("Échec de l'export du CSV", output)
        self.assertTrue(self.output_folder.is_dir())

    def test_filter_frames_ignores_unreadable_frame(self):
        good = self.write_frame("good.png")
        (self.input_folder / "invalid.png").write_text("This is not a valid image file.")

        frames, output = self.run_filter_frames()

        self.assertEqual(frames, [good])
        self.assertIn("Frame ignorée", output)

    def test_filter_frames_keeps_only_good_frames(self):
        good = self.write_frame("good.png")
        self.write_frame("burned.png", textured=False, saturated_percent=100)

        frames, output = self.run_filter_frames()

        self.assertEqual(frames, [good])
        self.assertIn("1/2 frames retenues", output)

    def test_filter_frames_csv_contains_rejected_frames(self):
        self.write_frame("burned.png", textured=False, saturated_percent=100)
        self.write_frame("good.png")
        csv_path = self.output_folder / "frames.csv"

        frames, _ = self.run_filter_frames(csv_path)

        # La frame brûlée n'est pas retournée, mais elle est dans le CSV avec ses scores
        rows = self.read_csv(csv_path)
        self.assertEqual(rows[0], ["frame_path", "sharpness", "darkness", "saturation", "result"])
        self.assertEqual(rows[1], ["burned.png", "0.0", "0.0", "100.0", "False"])
        self.assertEqual([row[0] for row in rows[1:]], ["burned.png", "good.png"])
        self.assertEqual([row[4] for row in rows[1:]], ["False", "True"])
        self.assertEqual(len(frames), 1)

    def test_filter_frames_csv_shows_saturation_of_overexposed_frame(self):
        self.write_frame("good.png")
        self.write_frame("overexposed.png", textured=False, saturated_percent=40)
        csv_path = self.output_folder / "frames.csv"

        frames, _ = self.run_filter_frames(csv_path)

        # La frame surexposée est rejetée, et le CSV montre pourquoi : 40 % de pixels saturés
        rows = self.read_csv(csv_path)
        self.assertEqual([row[0] for row in rows[1:]], ["good.png", "overexposed.png"])
        self.assertEqual([row[3] for row in rows[1:]], ["0.0", "40.0"])
        self.assertEqual([row[4] for row in rows[1:]], ["True", "False"])
        self.assertEqual(len(frames), 1)

    def test_filter_frames_with_subfolders(self):
        good = self.write_frame("video_01/frame_0001.png")
        csv_path = self.output_folder / "frames.csv"

        frames, _ = self.run_filter_frames(csv_path)

        self.assertEqual(frames, [good])
        self.assertEqual([row[0] for row in self.read_csv(csv_path)], ["frame_path", "video_01/frame_0001.png"])

    def test_filter_frames_with_nonexistent_input_folder(self):
        with self.assertRaises(ValueError):
            filter_frames(self.input_folder / "missing")