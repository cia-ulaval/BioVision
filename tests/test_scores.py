import os
import tempfile
import unittest
import numpy as np
import cv2

from pipeline.filtrage.scores import prepare_frame, calculate_scores, calculate_darkness_score, calculate_sharpness_score
from pipeline.filtrage.scores import calculate_saturation_score, calculate_contrast_score, calculate_brightness_score
from pipeline.filtrage import config

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
        at_threshold = np.full((100, 100), config.DARK_PIXEL_THRESHOLD, dtype=np.uint8)
        above_threshold = np.full((100, 100), config.DARK_PIXEL_THRESHOLD + 1, dtype=np.uint8)
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
        at_threshold = np.full((100, 100), config.SATURATED_PIXEL_THRESHOLD, dtype=np.uint8)
        below_threshold = np.full((100, 100), config.SATURATED_PIXEL_THRESHOLD - 1, dtype=np.uint8)
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

class TestCalculateContrastScore(unittest.TestCase):

    def test_contrast_score_with_uniform_image(self):
        # Une image uniforme n'a aucun contraste
        score = calculate_contrast_score(np.full((100, 100), 128, dtype=np.uint8))
        self.assertIsInstance(score, float)
        self.assertEqual(score, 0.0)

    def test_contrast_score_with_half_black_half_white_image(self):
        # Moitié 0, moitié 255 : écart-type de 127,5
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 255
        self.assertEqual(calculate_contrast_score(img), 127.5)

    def test_contrast_score_with_color_image(self):
        with self.assertRaises(ValueError):
            calculate_contrast_score(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_contrast_score_with_none_image(self):
        with self.assertRaises(ValueError):
            calculate_contrast_score(None)

class TestCalculateBrightnessScore(unittest.TestCase):

    def test_brightness_score_with_uniform_image(self):
        score = calculate_brightness_score(np.full((100, 100), 128, dtype=np.uint8))
        self.assertIsInstance(score, float)
        self.assertEqual(score, 128.0)

    def test_brightness_score_with_half_black_half_white_image(self):
        # Moitié 0, moitié 255 : moyenne de 127,5
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 255
        self.assertEqual(calculate_brightness_score(img), 127.5)

    def test_brightness_score_with_color_image(self):
        with self.assertRaises(ValueError):
            calculate_brightness_score(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_brightness_score_with_none_image(self):
        with self.assertRaises(ValueError):
            calculate_brightness_score(None)

class TestCalculateScores(unittest.TestCase):

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.temp_dir = temp_dir.name

    def write_image(self, img, name="image.png"):
        path = os.path.join(self.temp_dir, name)
        cv2.imwrite(path, img)
        return path

    def test_calculate_scores_with_uniform_image(self):
        # Aplat gris moyen : ni pixel noir ni saturé, aucune texture ni contraste
        path = self.write_image(np.full((100, 200), 128, dtype=np.uint8))

        self.assertEqual(calculate_scores(path), {
            "darkness": 0.0,
            "saturation": 0.0,
            "sharpness": 0.0,
            "contrast": 0.0,
            "brightness": 128.0,
        })

    def test_calculate_scores_matches_scores_of_prepared_image(self):
        # Image plus grande que MAX_SIDE : les scores doivent être ceux de l'image préparée (redimensionnée)
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(config.MAX_SIDE * 2, config.MAX_SIDE * 3), dtype=np.uint8)
        path = self.write_image(img)

        gray_img = prepare_frame(path)
        self.assertEqual(calculate_scores(path), {
            "darkness": calculate_darkness_score(gray_img),
            "saturation": calculate_saturation_score(gray_img),
            "sharpness": calculate_sharpness_score(gray_img),
            "contrast": calculate_contrast_score(gray_img),
            "brightness": calculate_brightness_score(gray_img),
        })

    def test_calculate_scores_with_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            calculate_scores(os.path.join(self.temp_dir, "nonexistent_file.png"))
