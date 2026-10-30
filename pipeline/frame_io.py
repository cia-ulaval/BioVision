from dataclasses import dataclass
from pathlib import Path
import csv

@dataclass
class FrameResult:
    """Résultat du filtrage d'une frame.

    Attributes:
        path (str or Path): Chemin de la frame.
        darkness (float): Score de noirceur (pourcentage de pixels noirs).
        saturation (float): Score de saturation (pourcentage de pixels saturés).
        sharpness (float): Score de netteté (variance du Laplacien).
        contrast (float): Score de contraste (écart-type des niveaux de gris).
        brightness (float): Score de luminosité (moyenne des niveaux de gris).
        result (bool): True si la frame est retenue, False sinon.
    """

    path: Path
    darkness: float
    saturation: float
    sharpness: float
    contrast: float
    brightness: float
    result: bool

def get_frames_from_folder(folder_path, extensions=(".jpg", ".png")):
    """Récupère les frames d'un dossier et de ses sous-dossiers selon les extensions spécifiées.

    Args:
        folder_path (str or Path): Chemin du dossier contenant les frames.
        extensions (tuple or str): Extensions de fichiers à inclure (par défaut : (".jpg", ".png")).

    Returns:
        list: Liste triée des chemins des frames correspondant aux extensions spécifiées,
            y compris celles des sous-dossiers.

    Raises:
        ValueError: Si folder_path n'existe pas ou n'est pas un dossier.
    """

    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Le chemin spécifié n'existe pas ou n'est pas un dossier : {folder_path}")

    # Vérifier si les extensions sont fournies sous forme de chaîne de caractères ou de tuple/list
    if isinstance(extensions, str):
        extensions = (extensions,)

    # Convertir les extensions en minuscules pour la comparaison et ajouter un point au début si nécessaire
    extensions = tuple(ext.lower() for ext in extensions)
    extensions = tuple(ext if ext.startswith('.') else f".{ext}" for ext in extensions)

    frames = []
    for element in folder.rglob("*"):
        if element.is_file() and element.suffix.lower() in extensions:
            frames.append(element)

    return sorted(frames)

def group_frames_by_video(frame_results):
    """Regroupe les résultats des frames par vidéo.

    La vidéo d'une frame est le dossier qui la contient (ex. ".../C00217_2021_03_12_193000_winter_20/f00012.jpg").

    Args:
        frame_results (list[FrameResult]): Résultats du filtrage des frames.

    Returns:
        dict[str, list[FrameResult]]: Nom de la vidéo -> ses FrameResult, dans l'ordre reçu.
    """

    frames_by_video = {}
    for frame_result in frame_results:
        video_name = Path(frame_result.path).parent.name
        frames_by_video.setdefault(video_name, []).append(frame_result)

    return frames_by_video

# Colonnes du CSV des résultats, dans l'ordre d'écriture
CSV_COLUMNS = ("frame_path", "sharpness", "darkness", "saturation", "contrast", "brightness", "result")

def export_frame_results_csv(frame_results, base_folder, csv_path):
    """Exporte les résultats du filtrage de toutes les frames vers un fichier CSV.

    Le CSV contient une ligne par frame avec les colonnes :
        - frame_path : chemin relatif à base_folder, toujours écrit avec des "/"
          (ex. : "video_01/frame_0001.jpg"). Pour retrouver une frame : base_folder / frame_path.
        - sharpness : score de netteté (variance du Laplacien).
        - darkness : score de noirceur (pourcentage de pixels noirs).
        - saturation : score de saturation (pourcentage de pixels saturés).
        - contrast : score de contraste (écart-type des niveaux de gris).
        - brightness : score de luminosité (moyenne des niveaux de gris).
        - result : True si la frame est retenue, False sinon.

    Args:
        frame_results (list[FrameResult]): Résultats à exporter.
        base_folder (str or Path): Dossier racine des frames.
        csv_path (str or Path): Chemin du fichier CSV de sortie.

    Returns:
        list[FrameResult]: Résultats écrits dans le CSV.

    Raises:
        ValueError: Si une frame ne se trouve pas dans base_folder.
    """

    base_folder = Path(base_folder)
    csv_path = Path(csv_path)

    parent_dir = csv_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(CSV_COLUMNS)

        for frame_result in frame_results:
            relative_path = Path(frame_result.path).relative_to(base_folder)
            csv_writer.writerow([
                relative_path.as_posix(),
                frame_result.sharpness,
                frame_result.darkness,
                frame_result.saturation,
                frame_result.contrast,
                frame_result.brightness,
                frame_result.result
            ])

    return frame_results

FILTER_MODES = ("all", "valid")

def read_frame_results_csv(csv_path, base_folder, filter_mode="all"):
    """Lit un CSV écrit par export_frame_results_csv et retourne les FrameResult.

    Args:
        csv_path (str or Path): Chemin du fichier CSV à lire.
        base_folder (str or Path): Dossier racine des frames, utilisé pour
            reconstruire leur chemin complet : base_folder / frame_path.
        filter_mode (str): "all" pour toutes les frames (par défaut),
            "valid" pour seulement les frames retenues.

    Returns:
        list[FrameResult]: Résultats lus, dans l'ordre du CSV.

    Raises:
        ValueError: Si filter_mode n'est pas une valeur de FILTER_MODES.
        ValueError: Si une colonne de CSV_COLUMNS manque (CSV d'une ancienne version du filtre).
        FileNotFoundError: Si le CSV n'existe pas.
    """

    # Vérifie que le filter_mode est valide
    if filter_mode not in FILTER_MODES:
        raise ValueError(f"filter_mode invalide : {filter_mode!r} (attendu : {FILTER_MODES})")

    base_folder = Path(base_folder)

    frame_results = []
    with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        # Vérifie que le CSV contient toutes les colonnes attendues
        missing_columns = [column for column in CSV_COLUMNS if column not in (reader.fieldnames or [])]
        if missing_columns:
            raise ValueError(
                f"Colonnes manquantes dans {csv_path} : {missing_columns}. "
                "Ce CSV vient probablement d'une ancienne version du filtre : relancer filter_frames pour le régénérer."
            )

        for row in reader:
            # Lit le résultat de la frame
            result = row["result"] == "True"

            # Ignore la frame si elle est rejetée et que filter_mode vaut "valid"
            if filter_mode == "valid" and not result:
                continue

            frame_results.append(FrameResult(
                path=base_folder / row["frame_path"],
                darkness=float(row["darkness"]),
                saturation=float(row["saturation"]),
                sharpness=float(row["sharpness"]),
                contrast=float(row["contrast"]),
                brightness=float(row["brightness"]),
                result=result
            ))

    return frame_results