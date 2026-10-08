from pathlib import Path
import cv2
import numpy as np

# Récupération des frames en utilisant un chemin d'accès à un dossier contenant les frames
PATH_TO_FRAMES = Path(__file__).parent.parent / "data" / "frames_test" 

# Taille maximale du côté pour le redimensionnement des frames
MAX_SIDE = 640  

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
    scale_factor = MAX_SIDE / max(height, width)

    if scale_factor >= 1:
        # L'image est plus petite ou égale à MAX_SIDE, on peut la garder telle quelle
        return img

    new_height = max(1, int(height * scale_factor))
    new_width = max(1, int(width * scale_factor))
    img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return img_resized

# Flou : variance laplacien en dessous de ce seuil est considéré comme flou
SHARPNESS_THRESHOLD = 40.0

# Exposition : luminosité moyenne en dehors de ce seuil est considérée comme sous-exposée ou surexposée
EXPOSURE_THRESHOLD_LOW = 55.0
EXPOSURE_THRESHOLD_HIGH = 160.0

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

def calculate_exposure_score(gray_img):
    """Calcule le score d'exposition d'une image en niveaux de gris en utilisant la luminosité moyenne entre les seuils (EXPOSURE_THRESHOLD_LOW, EXPOSURE_THRESHOLD_HIGH).

    Args:
        gray_img (np.ndarray): Image en niveaux de gris.

    Raises:
        ValueError: Si l'image n'est pas en niveaux de gris (2 dimensions).
        ValueError: Si l'image est None.

    Returns:    
        float: Score d'exposition (luminosité moyenne).
    """

    if gray_img is None:
        raise ValueError("L'image ne peut pas être None.")

    if gray_img.ndim != 2:
        raise ValueError("L'image doit être en niveaux de gris (2 dimensions).")

    return float(np.mean(gray_img))

def validate_frame_quality(frame):
    """Valide la qualité d'une frame en vérifiant son exposition et sa netteté.

    Args:
        frame (str or Path): Chemin de la frame à valider.

    Returns:
        bool: True si la frame est de bonne qualité, False sinon.
    """

    gray_img = prepare_frame(frame)
    exposure = calculate_exposure_score(gray_img)
    sharpness = calculate_sharpness_score(gray_img)

    if sharpness is None or exposure is None:
        return False

    if exposure < EXPOSURE_THRESHOLD_LOW or exposure > EXPOSURE_THRESHOLD_HIGH:
        return False

    if sharpness < SHARPNESS_THRESHOLD:
        return False

    return True