from pathlib import Path
import cv2
import numpy as np

from pipeline.frame_io import get_frames_from_folder, export_frame_results_csv, FrameResult

# Récupération des frames en utilisant un chemin d'accès à un dossier contenant les frames
PATH_TO_FRAMES = Path(__file__).parent.parent / "data" / "frames" / "caribou"

# Chemin pour le csv des résultats
PATH_TO_CSV = Path(__file__).parent.parent / "data" / "results" / "result.csv"

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

# Un pixel jusqu'à ce niveau de gris est considéré comme noir (les caméras descendent vers 2-3, pas 0)
DARK_PIXEL_THRESHOLD = 10
# Une frame dont le pourcentage de pixels noirs dépasse ce seuil est considérée comme sous-exposée
DARK_PERCENT_THRESHOLD = 25.0

# Un pixel à partir de ce niveau de gris est considéré comme saturé (les caméras plafonnent vers 246, pas 255)
SATURATED_PIXEL_THRESHOLD = 240
# Une frame dont le pourcentage de pixels saturés dépasse ce seuil est considérée comme surexposée
SATURATED_PERCENT_THRESHOLD = 25.0

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

    dark_pixels = np.count_nonzero(gray_img <= DARK_PIXEL_THRESHOLD)
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

    saturated_pixels = np.count_nonzero(gray_img >= SATURATED_PIXEL_THRESHOLD)
    return float(100 * saturated_pixels / gray_img.size)

def validate_frame_quality(frame):
    """Valide la qualité d'une frame en vérifiant sa noirceur, sa saturation et sa netteté.

    Args:
        frame (str or Path): Chemin de la frame à valider.

    Returns:
        FrameResult: Chemin de la frame, scores calculés (darkness, saturation, sharpness)
            et result qui vaut True si la frame est de bonne qualité, False sinon.
    """

    gray_img = prepare_frame(frame)
    darkness = calculate_darkness_score(gray_img)
    saturation = calculate_saturation_score(gray_img)
    sharpness = calculate_sharpness_score(gray_img)

    result = (
        darkness <= DARK_PERCENT_THRESHOLD
        and saturation <= SATURATED_PERCENT_THRESHOLD
        and sharpness >= SHARPNESS_THRESHOLD
    )

    return FrameResult(path=frame, darkness=darkness, saturation=saturation, sharpness=sharpness, result=result)

def filter_frames(input_folder=PATH_TO_FRAMES, csv_path=PATH_TO_CSV):
    """Filtre les frames d'un dossier (et de ses sous-dossiers) et exporte les
    résultats de toutes les frames dans un CSV.

    Les frames ne sont pas copiées : le CSV contient leurs chemins relatifs à
    input_folder, leurs scores de netteté, de noirceur et de saturation, et la décision.

    Un message indique toujours le résultat de l'export : CSV écrit, aucun export
    demandé (csv_path vaut None) ou échec de l'écriture. En cas d'échec, les frames
    valides sont quand même retournées.

    Args:
        input_folder: Dossier source (par défaut : PATH_TO_FRAMES).
        csv_path: Chemin du fichier CSV de sortie (par défaut : PATH_TO_CSV).
            Si None, pas d'export.

    Returns:
        list: Chemins des frames valides.
    """

    frames = get_frames_from_folder(input_folder)

    frame_results = []
    for frame in frames:
        try:
            frame_results.append(validate_frame_quality(frame))
        except FileNotFoundError as error:
            # Image corrompue ou illisible : on l'ignore et on continue
            print(f"Frame ignorée : {error}")

    valid_frames = [frame_result.path for frame_result in frame_results if frame_result.result]

    print(f"{len(valid_frames)}/{len(frames)} frames retenues")

    if csv_path is None:
        print("Aucun CSV exporté : csv_path non fourni")
    else:
        try:
            export_frame_results_csv(frame_results, input_folder, csv_path)
            print(f"CSV exporté : {csv_path}")
        except OSError as error:
            # Problème d'écriture
            print(f"Échec de l'export du CSV : {error}")

    return valid_frames