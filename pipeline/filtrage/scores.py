import cv2
import numpy as np

from pipeline.filtrage import config

def prepare_frame(frame):
    """Prépare une frame pour le filtrage en la redimensionnant et en la convertissant en niveaux de gris.
    
    Args:
        frame (str or Path): Chemin de la frame à préparer.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas ou ne peut pas être lu.
        
    Returns:
        np.ndarray: Image préparée en niveaux de gris.
    """

    # lire l'image en niveaux de gris
    img = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Impossible de lire l'image à partir du chemin : {frame}")

    height, width = img.shape

    # Calculer le facteur d'échelle pour redimensionner l'image si nécessaire
    scale_factor = config.MAX_SIDE / max(height, width)

    if scale_factor >= 1:
        # L'image est plus petite ou égale à MAX_SIDE, on peut la garder telle quelle
        return img

    new_height = max(1, int(height * scale_factor))
    new_width = max(1, int(width * scale_factor))
    img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return img_resized

def calculate_sharpness_score(gray_img):
    """Calcule le score de netteté d'une image en niveaux de gris en utilisant la variance du Laplacien.

    Args:
        gray_img (np.ndarray): Image en niveaux de gris.

    Raises:
        ValueError: Si l'image n'est pas en niveaux de gris (2 dimensions).
        ValueError: Si l'image est None.

    Returns:
        float: Score de netteté (variance du Laplacien).
    """

    if gray_img is None:
        raise ValueError("L'image ne peut pas être None.")

    # Vérifier que l'image est en niveaux de gris (2 dimensions)
    if gray_img.ndim != 2:
        raise ValueError("L'image doit être en niveaux de gris (2 dimensions).")

    laplacian_var = float(cv2.Laplacian(gray_img, cv2.CV_64F).var())
    return laplacian_var

def calculate_darkness_score(gray_img):
    """Calcule le pourcentage de pixels noirs (<= DARK_PIXEL_THRESHOLD) d'une image en niveaux de gris, comparé à DARK_PERCENT_THRESHOLD.

    Args:
        gray_img (np.ndarray): Image en niveaux de gris.

    Raises:
        ValueError: Si l'image n'est pas en niveaux de gris (2 dimensions).
        ValueError: Si l'image est None.

    Returns:
        float: Score de noirceur (pourcentage de pixels noirs, entre 0 et 100).
    """

    if gray_img is None:
        raise ValueError("L'image ne peut pas être None.")

    if gray_img.ndim != 2:
        raise ValueError("L'image doit être en niveaux de gris (2 dimensions).")

    dark_pixels = np.count_nonzero(gray_img <= config.DARK_PIXEL_THRESHOLD)
    return float(100 * dark_pixels / gray_img.size)

def calculate_saturation_score(gray_img):
    """Calcule le pourcentage de pixels saturés (>= SATURATED_PIXEL_THRESHOLD) d'une image en niveaux de gris, comparé à SATURATED_PERCENT_THRESHOLD.

    Args:
        gray_img (np.ndarray): Image en niveaux de gris.

    Raises:
        ValueError: Si l'image n'est pas en niveaux de gris (2 dimensions).
        ValueError: Si l'image est None.

    Returns:
        float: Score de saturation (pourcentage de pixels saturés, entre 0 et 100).
    """

    if gray_img is None:
        raise ValueError("L'image ne peut pas être None.")

    if gray_img.ndim != 2:
        raise ValueError("L'image doit être en niveaux de gris (2 dimensions).")

    saturated_pixels = np.count_nonzero(gray_img >= config.SATURATED_PIXEL_THRESHOLD)
    return float(100 * saturated_pixels / gray_img.size)

def calculate_contrast_score(gray_img):
    """Calcule le contraste d'une image en niveaux de gris : l'écart-type de ses niveaux de gris.

    Args:
        gray_img (np.ndarray): Image en niveaux de gris.

    Raises:
        ValueError: Si l'image n'est pas en niveaux de gris (2 dimensions).
        ValueError: Si l'image est None.

    Returns:
        float: Score de contraste (0 pour une image uniforme).
    """

    if gray_img is None:
        raise ValueError("L'image ne peut pas être None.")

    if gray_img.ndim != 2:
        raise ValueError("L'image doit être en niveaux de gris (2 dimensions).")

    return float(gray_img.std())

def calculate_brightness_score(gray_img):
    """Calcule la luminosité d'une image en niveaux de gris : la moyenne de ses niveaux de gris.
    Args:
        gray_img (np.ndarray): Image en niveaux de gris.

    Raises:
        ValueError: Si l'image n'est pas en niveaux de gris (2 dimensions).
        ValueError: Si l'image est None.

    Returns:
        float: Score de luminosité (entre 0 pour noir et 255 pour blanc).
    """

    if gray_img is None:
        raise ValueError("L'image ne peut pas être None.")

    if gray_img.ndim != 2:
        raise ValueError("L'image doit être en niveaux de gris (2 dimensions).")

    return float(gray_img.mean())

def calculate_scores(frame):
    """Calcule tous les scores d'une frame, préparée une seule fois avec prepare_frame.

    Point d'entrée à utiliser depuis un autre module : il garantit que les scores
    sont calculés sur une image préparée comme pendant la calibration des seuils
    (niveaux de gris, côté max de config.MAX_SIDE).

    Args:
        frame (str or Path): Chemin de la frame.

    Returns:
        dict[str, float]: Clés "darkness", "saturation", "sharpness", "contrast", "brightness".

    Raises:
        FileNotFoundError: Si la frame n'existe pas ou ne peut pas être lue (levée par prepare_frame).
    """

    gray_img = prepare_frame(frame)

    return {
        "darkness": calculate_darkness_score(gray_img),
        "saturation": calculate_saturation_score(gray_img),
        "sharpness": calculate_sharpness_score(gray_img),
        "contrast": calculate_contrast_score(gray_img),
        "brightness": calculate_brightness_score(gray_img),
    }