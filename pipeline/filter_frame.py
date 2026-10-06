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

    Returns:
        np.ndarray: Image préparée en niveaux de gris.
        None: Si l'image ne peut pas être lue ou est invalide.
    """

    # lire l'image en niveaux de gris
    img = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

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
SHARPNESS_THRESHOLD = 100.0

# Exposition : luminosité moyenne en dehors de ce seuil est considérée comme sous-exposée ou surexposée
EXPOSURE_THRESHOLD_LOW = 40.0
EXPOSURE_THRESHOLD_HIGH = 220.0

def calculate_sharpness_score(gray_img):
    """Calcule le score de netteté d'une image en niveaux de gris en utilisant la variance du Laplacien.

    Args:
        gray_img (np.ndarray): Image en niveaux de gris.

    Raises:
        ValueError: Si l'image n'est pas en niveaux de gris (2 dimensions).

    Returns:
        float: Score de netteté (variance du Laplacien).
        None: Si l'image est None.
    """
    if gray_img is None:
        return None

    # Vérifier que l'image est en niveaux de gris (2 dimensions)
    if gray_img.ndim != 2:
        raise ValueError("L'image doit être en niveaux de gris (2 dimensions).")

    laplacian_var = float(cv2.Laplacian(gray_img, cv2.CV_64F).var())
    return laplacian_var

# Créer une fonction pour calculer le score d'exposition d'une frame
def calculate_exposure_score(frame):
    return None  # Remplacer par le code pour calculer le score d'exposition;

# Créer une fonction pour calculer le score global d'une frame en combinant les scores de netteté et d'exposition
def calculate_overall_score(sharpness_score, exposure_score):
    return None  # Remplacer par le code pour calculer le score global;

# Créer une fonction pour prendre une décision sur la qualité d'une frame en fonction de ses scores
def decide_frame_quality(sharpness_score, exposure_score):
    return None  # Remplacer par le code pour décider de la qualité de la frame;