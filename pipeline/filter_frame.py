from pathlib import Path
import cv2
import numpy as np

from pipeline.frame_io import get_frames_from_folder, export_frame_results_csv, group_frames_by_video, FrameResult

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

# Texture : une variance du Laplacien en dessous de ce seuil est considérée comme un aplat sans texture.
# Ne sert jamais seul (une frame floue reste gardée) : seulement combiné à la saturation ou à la noirceur.
# Valeur provisoire, à calibrer avec la colonne NotUsable des biologistes
TEXTURE_THRESHOLD = 5.0

# Un pixel jusqu'à ce niveau de gris est considéré comme noir (les caméras descendent vers 2-3, pas 0)
DARK_PIXEL_THRESHOLD = 10
# Une frame dont le pourcentage de pixels noirs dépasse ce seuil est considérée comme sous-exposée
DARK_PERCENT_THRESHOLD = 25.0

# Un pixel à partir de ce niveau de gris est considéré comme saturé (les caméras plafonnent vers 246, pas 255)
SATURATED_PIXEL_THRESHOLD = 240
# Une frame dont le pourcentage de pixels saturés dépasse ce seuil est considérée comme surexposée
SATURATED_PERCENT_THRESHOLD = 25.0

# Cadence d'extraction des frames
FRAMES_PER_SECOND = 1
# Une vidéo dont la durée ne dépasse pas ce seuil (en secondes) est rejetée en entier.
MIN_VIDEO_DURATION = 1.0

# Voile (buée, givre) : une frame est voilée si son contraste ET sa texture sont bas ET sa luminosité est moyenne.
# Contraste (écart-type des niveaux de gris) en dessous duquel l'image est un aplat uniforme
VEIL_CONTRAST_THRESHOLD = 12.0
# Texture en dessous de laquelle un aplat peut être un voile.
VEIL_TEXTURE_THRESHOLD = 20.0
# Luminosité (moyenne des niveaux de gris) d'un voile. En dessous : scène de nuit, peu contrastée mais utilisable.
# Au-dessus : frame brûlée, déjà gérée par SATURATED_PERCENT_THRESHOLD
VEIL_BRIGHTNESS_MIN = 60.0
VEIL_BRIGHTNESS_MAX = 230.0

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

def validate_frame_quality(frame):
    """Valide la qualité d'une frame en vérifiant sa noirceur, sa saturation, sa netteté, son contraste et sa luminosité.

    Une frame est rejetée si elle est :
        - brûlée : trop de pixels saturés (> SATURATED_PERCENT_THRESHOLD) ET sans texture (< TEXTURE_THRESHOLD) ;
        - noire : trop de pixels noirs (> DARK_PERCENT_THRESHOLD) ET sans texture (< TEXTURE_THRESHOLD) ;
        - voilée (buée, givre) : contraste bas (< VEIL_CONTRAST_THRESHOLD) ET texture basse (< VEIL_TEXTURE_THRESHOLD)
          ET luminosité moyenne (entre VEIL_BRIGHTNESS_MIN et VEIL_BRIGHTNESS_MAX).
    Le flou seul ne la rejette jamais : chaque condition combine plusieurs mesures.

    Args:
        frame (str or Path): Chemin de la frame à valider.

    Returns:
        FrameResult: Chemin de la frame, scores calculés (darkness, saturation, sharpness, contrast, brightness)
            et result qui vaut True si la frame est de bonne qualité, False sinon.

    Raises:
        FileNotFoundError: Si la frame n'existe pas ou ne peut pas être lue (levée par prepare_frame).
    """

    gray_img = prepare_frame(frame)
    darkness = calculate_darkness_score(gray_img)
    saturation = calculate_saturation_score(gray_img)
    sharpness = calculate_sharpness_score(gray_img)
    contrast = calculate_contrast_score(gray_img)
    brightness = calculate_brightness_score(gray_img)

    has_texture = sharpness >= TEXTURE_THRESHOLD
    is_burned = (saturation > SATURATED_PERCENT_THRESHOLD) and not has_texture
    is_dark = darkness > DARK_PERCENT_THRESHOLD and not has_texture
    is_veil = (
        contrast < VEIL_CONTRAST_THRESHOLD
        and sharpness < VEIL_TEXTURE_THRESHOLD
        and VEIL_BRIGHTNESS_MIN <= brightness <= VEIL_BRIGHTNESS_MAX
    )

    result = (
        not is_burned and not is_dark and not is_veil
    )

    return FrameResult(path=frame, darkness=darkness, saturation=saturation, sharpness=sharpness,
                       contrast=contrast, brightness=brightness, result=result)

def reject_short_videos(frame_results):
    """Rejette toutes les frames des vidéos trop courtes.

    Une vidéo est trop courte si son nombre de frames ne dépasse pas
    FRAMES_PER_SECOND * MIN_VIDEO_DURATION. Les frames sont regroupées par vidéo
    avec group_frames_by_video, et le result de chaque frame d'une vidéo trop
    courte passe à False, quels que soient ses scores.

    Args:
        frame_results (list[FrameResult]): Résultats de validate_frame_quality.

    Returns:
        list[FrameResult]: Les mêmes résultats, modifiés sur place.
    """

    for video_frames in group_frames_by_video(frame_results).values():
        if len(video_frames) <= FRAMES_PER_SECOND * MIN_VIDEO_DURATION:
            for frame in video_frames:
                frame.result = False

    return frame_results

def filter_frames(input_folder=PATH_TO_FRAMES, csv_path=PATH_TO_CSV):
    """Filtre les frames d'un dossier (et de ses sous-dossiers) et exporte les
    résultats de toutes les frames dans un CSV.

    Les frames ne sont pas copiées : le CSV contient leurs chemins relatifs à
    input_folder, leurs scores de netteté, de noirceur et de saturation, et la décision.

    Une frame est rejetée si elle est de mauvaise qualité (validate_frame_quality)
    ou si sa vidéo est trop courte (reject_short_videos). Chaque sous-dossier
    d'input_folder est une vidéo.

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

    reject_short_videos(frame_results)

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