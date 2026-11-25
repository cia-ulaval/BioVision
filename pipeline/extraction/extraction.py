from pathlib import Path
from pipeline.extraction.config import PATH_TO_ANNOTATIONS

def trouver_csv_du_lot(dossier_lot, dossier_annotations=PATH_TO_ANNOTATIONS):
    """Trouve le CSV Timelapse d'un lot annoté.

    Le CSV est cherché directement dans le dossier du lot, sans descendre dans les
    sous-dossiers (pour ignorer les copies de sauvegarde de Timelapse dans Backups/).

    Args:
        dossier_lot (str or Path): Chemin du lot, relatif au dossier des annotations
            (ex. "to_analyse2" ou "to_analyse/Caribou_rand_Fanny1").
        dossier_annotations (str or Path): Dossier qui contient les lots
            (par défaut : PATH_TO_ANNOTATIONS).

    Returns:
        Path: Chemin du seul CSV du lot.

    Raises:
        FileNotFoundError: Si le lot n'existe pas ou ne contient aucun CSV.
        ValueError: Si le lot contient plusieurs CSV.
    """

    chemin = Path(dossier_annotations) / dossier_lot

    if not chemin.is_dir():
        raise FileNotFoundError(f"Le lot n'existe pas ou n'est pas un dossier : {chemin}")

    fichiers = sorted(chemin.glob("*.csv"))

    if len(fichiers) == 0:
        raise FileNotFoundError(f"Aucun CSV dans le lot : {chemin}")
    elif len(fichiers) > 1:
        noms = ", ".join(fichier.name for fichier in fichiers)
        raise ValueError(f"Plusieurs CSV dans le lot {chemin} : {noms}")

    return fichiers[0]

def lire_videos_du_lot(csv_path):
    return NotImplemented

def dossier_frames_de_video(chemin_video):
    return NotImplemented

def extraire_frames_video(chemin_video, dossier_sortie):
    return NotImplemented