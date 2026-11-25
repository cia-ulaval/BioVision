import csv
import warnings
from pathlib import Path

import cv2
from tqdm.auto import tqdm

from pipeline.extraction.config import PATH_TO_ANNOTATIONS, PATH_TO_FRAMES, FRAMES_PER_SECOND, LOTS_ANNOTES

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

def extraire_frames_video(chemin_video, dossier_sortie, frames_per_second=FRAMES_PER_SECOND):
    """Extrait les frames d'une vidéo à une cadence fixe et les écrit en .jpg.

    Chaque frame est nommée d'après son index dans la vidéo (ex. f00000.jpg, f00015.jpg),
    comme les frames déjà extraites. Une vidéo déjà extraite (dossier contenant des .jpg)
    n'est pas extraite à nouveau.

    Args:
        chemin_video (str or Path): Chemin de la vidéo.
        dossier_sortie (str or Path): Dossier où écrire les frames (créé au besoin).
        frames_per_second (float): Nombre de frames gardées par seconde de vidéo
            (par défaut : FRAMES_PER_SECOND).

    Returns:
        list[Path]: Chemins triés des frames de la vidéo. Liste vide si la vidéo est illisible.
    """

    chemin_video = Path(chemin_video)
    dossier_sortie = Path(dossier_sortie)

    frames_existantes = sorted(dossier_sortie.glob("*.jpg"))
    if frames_existantes:
        return frames_existantes

    capture = cv2.VideoCapture(str(chemin_video))
    try:
        if not capture.isOpened():
            warnings.warn(f"Vidéo illisible, ignorée : {chemin_video}")
            return []

        fps_video = capture.get(cv2.CAP_PROP_FPS)
        if fps_video <= 0:
            warnings.warn(f"Cadence inconnue, vidéo ignorée : {chemin_video}")
            return []

        # Ex. vidéo à 15 fps et 1 frame par seconde : on garde une frame sur 15
        pas = max(1, round(fps_video / frames_per_second))

        dossier_sortie.mkdir(parents=True, exist_ok=True)
        frames = []
        index = 0

        # grab() avance sans décoder : seules les frames gardées sont décodées avec retrieve()
        while capture.grab():
            if index % pas == 0:
                lue, image = capture.retrieve()
                if lue:
                    chemin_frame = dossier_sortie / f"f{index:05d}.jpg"
                    cv2.imwrite(str(chemin_frame), image)
                    frames.append(chemin_frame)

            index += 1
    finally:
        capture.release()

    return frames

def extraire_frames_des_lots(lots=LOTS_ANNOTES, dossier_annotations=PATH_TO_ANNOTATIONS, dossier_frames=PATH_TO_FRAMES):
    """Extrait les frames de toutes les vidéos annotées des lots, rangées par caribou puis par vidéo.

    Les vidéos déjà extraites ne sont pas extraites à nouveau : l'extraction peut être relancée
    après une interruption. Un bilan est affiché à la fin.

    Args:
        lots (list[str]): Lots à extraire, relatifs au dossier des annotations (par défaut : LOTS_ANNOTES).
        dossier_annotations (str or Path): Dossier qui contient les lots (par défaut : PATH_TO_ANNOTATIONS).
        dossier_frames (str or Path): Dossier racine des frames (par défaut : PATH_TO_FRAMES).

    Returns:
        dict[str, list[Path]]: Nom de la vidéo -> chemins de ses frames.

    Raises:
        FileNotFoundError, ValueError: Si un lot n'existe pas ou n'a pas exactement un CSV.
    """

    frames_par_video = {}

    for lot in lots:
        videos = lire_videos_du_lot(trouver_csv_du_lot(lot, dossier_annotations))

        for nom_video, chemin_video in tqdm(videos.items(), desc=lot):
            if nom_video in frames_par_video:
                warnings.warn(f"Vidéo déjà présente dans un autre lot, ignorée : {chemin_video}")
                continue

            dossier_sortie = dossier_frames_de_video(chemin_video, dossier_frames)
            frames_par_video[nom_video] = extraire_frames_video(chemin_video, dossier_sortie)

    videos_vides = [nom for nom, frames in frames_par_video.items() if not frames]
    nb_frames = sum(len(frames) for frames in frames_par_video.values())
    print(f"{len(frames_par_video)} vidéos, {nb_frames} frames, {len(videos_vides)} vidéos sans frame")

    return frames_par_video