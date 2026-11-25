import csv
import warnings
from pathlib import Path
from pipeline.extraction.config import PATH_TO_ANNOTATIONS, PATH_TO_FRAMES

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
    """Lit le CSV Timelapse d'un lot et retrouve le chemin de chaque vidéo annotée.

    Le chemin est construit à partir du dossier du CSV (les vidéos sont à côté de leur CSV).
    Une vidéo absente du disque ou en double dans le CSV est ignorée avec un avertissement.

    Args:
        csv_path (str or Path): Chemin du CSV Timelapse du lot.

    Returns:
        dict[str, Path]: Nom de la vidéo sans extension -> chemin de la vidéo.
    """

    csv_path = Path(csv_path)
    videos = {}

    with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            fichier = Path(row["File"])
            video_name = fichier.stem
            # Timelapse tourne sous Windows : les sous-dossiers sont séparés par des \
            relative_path = row["RelativePath"].replace("\\", "/")
            chemin_video = csv_path.parent / relative_path / fichier

            if video_name in videos:
                warnings.warn(f"Vidéo en double dans {csv_path.name}, ignorée : {fichier}")
                continue

            if not chemin_video.exists():
                warnings.warn(f"Vidéo introuvable, ignorée : {chemin_video}")
                continue

            videos[video_name] = chemin_video

    return videos

def dossier_frames_de_video(chemin_video, dossier_frames=PATH_TO_FRAMES):
    """Construit le dossier de sortie des frames d'une vidéo : <dossier_frames>/<caribou>/<vidéo>/.

    Le dossier n'est pas créé : seul le chemin est calculé.

    Args:
        chemin_video (str or Path): Chemin de la vidéo (ex. ".../C00217_2021_03_12_193000_winter_20.avi").
        dossier_frames (str or Path): Dossier racine des frames.

    Returns:
        Path: Dossier des frames de la vidéo (ex. "/frames/C00217/C00217_2021_03_12_193000_winter_20").
    """

    nom_video = Path(chemin_video).stem
    # Le nom de la vidéo commence par l'identifiant du caribou (ex. "C00217")
    caribou = nom_video[:6]

    return Path(dossier_frames) / caribou / nom_video

def extraire_frames_video(chemin_video, dossier_sortie):
    return NotImplemented