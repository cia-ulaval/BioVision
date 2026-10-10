import os
import tempfile
import unittest
import numpy as np
import cv2

from pipeline.filter_frame import prepare_frame, validate_frame_quality, calculate_exposure_score, calculate_sharpness_score
from pipeline.filter_frame import EXPOSURE_THRESHOLD_LOW, EXPOSURE_THRESHOLD_HIGH

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

class TestCalculateExposureScore(unittest.TestCase):

    def test_exposure_score_with_uniform_image(self):
        # Une image uniforme (tous les pixels ont la même valeur) doit avoir un score d'exposition égal à la valeur des pixels
        uniform_img = np.full((100, 100), 128, dtype=np.uint8)
        score = calculate_exposure_score(uniform_img)
        self.assertIsInstance(score, float)
        self.assertEqual(score, 128.0)

    def test_exposure_score_with_black_image(self):
        # Une image noire doit avoir le score minimal (0)
        black_img = np.zeros((100, 100), dtype=np.uint8)
        self.assertEqual(calculate_exposure_score(black_img), 0.0)

    def test_exposure_score_with_white_image(self):
        # Une image blanche doit avoir le score maximal (255)
        white_img = np.full((100, 100), 255, dtype=np.uint8)
        self.assertEqual(calculate_exposure_score(white_img), 255.0)

    def test_exposure_score_with_half_black_half_white_image(self):
        # Moitié noire (0), moitié blanche (255) : la moyenne doit être 127.5
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 255
        self.assertEqual(calculate_exposure_score(img), 127.5)

    def test_exposure_score_with_color_image(self):
        # Tester que la fonction lève une exception pour une image couleur
        color_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            calculate_exposure_score(color_img)

    def test_exposure_score_with_none_image(self):
        # Tester que la fonction lève une exception pour une image None
        with self.assertRaises(ValueError):
            calculate_exposure_score(None)

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

    def make_checkerboard_with_mean(self, mean):
        # Damier net dont la moyenne vaut exactement `mean` (contraste maximal possible)
        high = min(255, int(2 * mean))
        low = int(2 * mean) - high
        return self.make_checkerboard(low, high)

    def test_validate_frame_quality_with_good_image(self):
        # Nette (damier) et bien exposée (moyenne 127.5)
        path = self.write_image(self.make_checkerboard(0, 255))
        self.assertTrue(validate_frame_quality(path))

    def test_validate_frame_quality_with_blurry_image(self):
        # Bien exposée (moyenne 128) mais floue (aucun contour, netteté 0)
        path = self.write_image(np.full((100, 100), 128, dtype=np.uint8))
        self.assertFalse(validate_frame_quality(path))

    def test_validate_frame_quality_with_underexposed_image(self):
        # Nette mais trop sombre (moyenne 10 sous le seuil bas)
        path = self.write_image(self.make_checkerboard_with_mean(EXPOSURE_THRESHOLD_LOW - 10))
        self.assertFalse(validate_frame_quality(path))

    def test_validate_frame_quality_with_overexposed_image(self):
        # Nette mais trop claire (moyenne 10 au-dessus du seuil haut)
        path = self.write_image(self.make_checkerboard_with_mean(EXPOSURE_THRESHOLD_HIGH + 10))
        self.assertFalse(validate_frame_quality(path))

    def test_validate_frame_quality_at_exposure_thresholds(self):
        # Les seuils eux-mêmes sont acceptés (comparaisons strictes < et >)
        for mean in (EXPOSURE_THRESHOLD_LOW, EXPOSURE_THRESHOLD_HIGH):
            with self.subTest(mean=mean):
                img = self.make_checkerboard_with_mean(mean)
                self.assertEqual(calculate_exposure_score(img), mean)
                path = self.write_image(img)
                self.assertTrue(validate_frame_quality(path))

    def test_validate_frame_quality_with_color_image(self):
        # Une image couleur est convertie en niveaux de gris : c'est une entrée valide
        color_img = cv2.cvtColor(self.make_checkerboard(0, 255), cv2.COLOR_GRAY2BGR)
        path = self.write_image(color_img)
        self.assertTrue(validate_frame_quality(path))

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