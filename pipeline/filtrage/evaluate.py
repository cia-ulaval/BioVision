import csv
from pathlib import Path

from pipeline.filtrage.frame_io import group_frames_by_video
from pipeline.filtrage import config

def read_not_usable_labels(csv_path):
    """Lit le fichier d'annotation des biologistes et retourne leur décision NotUsable pour chaque vidéo.

    Args:
        csv_path (str or Path): Chemin du CSV d'annotation (colonnes "File" et "NotUsable").

    Returns:
        dict[str, bool]: Nom de la vidéo sans extension (ex. "C00217_2021_03_12_193000_winter_20")
            -> True si les biologistes l'ont jugée NotUsable, False sinon.

    Raises:
        FileNotFoundError: Si le CSV n'existe pas.
    """

    not_usable_by_video = {}

    with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            video_name = Path(row["File"]).stem
            not_usable_by_video[video_name] = row["NotUsable"].strip().lower() == "true"

    return not_usable_by_video

def video_rejection_percents(frame_results):
    """Calcule le pourcentage de frames rejetées de chaque vidéo.

    La vidéo d'une frame est le dossier qui la contient (ex. ".../C00217_2021_03_12_193000_winter_20/f00012.jpg").

    Args:
        frame_results (list[FrameResult]): Résultats du filtrage des frames.

    Returns:
        dict[str, float]: Nom de la vidéo -> pourcentage de ses frames rejetées (entre 0 et 100).
    """

    rejection_percents = {}
    for video_name, video_frames in group_frames_by_video(frame_results).items():
        rejected_count = sum(not frame_result.result for frame_result in video_frames)
        rejection_percents[video_name] = 100 * rejected_count / len(video_frames)

    return rejection_percents

def predict_not_usable(rejection_percents):
    """Prédit pour chaque vidéo si elle est NotUsable à partir de son pourcentage de frames rejetées.

    Args:
        rejection_percents (dict[str, float]): Résultat de video_rejection_percents.

    Returns:
        dict[str, bool]: Nom de la vidéo -> True si son pourcentage dépasse VIDEO_REJECT_PERCENT.
    """

    return {video_name: percent > config.VIDEO_REJECT_PERCENT for video_name, percent in rejection_percents.items()}

def calculate_recall(predictions, labels):
    """Calcule le rappel du filtre par rapport aux étiquettes NotUsable des biologistes.

    Répond à la question : le filtre oublie-t-il des vidéos que les biologistes ont rejetées ?
    Rappel = vrais positifs / (vrais positifs + faux négatifs).

    Proche de 1 : le filtre attrape presque toutes les vidéos NotUsable.
    Proche de 0 : il en laisse passer la plupart.
    (à lire avec la précision : rejeter tout donne aussi 1)

    Args:
        predictions (dict[str, bool]): Résultat de predict_not_usable.
        labels (dict[str, bool]): Résultat de read_not_usable_labels.

    Returns:
        float | None: Rappel entre 0 et 1, ou None si aucune vidéo NotUsable n'est comptée.
    """

    # Vidéos jugées NotUsable par les biologistes
    not_usable_videos = {video_name for video_name, is_not_usable in labels.items() if is_not_usable}

    # Vrais positifs : rejetées par le filtre et par les biologistes
    # Faux négatifs : gardées par le filtre, mais rejetées par les biologistes
    true_positives = 0
    false_negatives = 0

    for video_name, predicted_not_usable in predictions.items():
        if predicted_not_usable and video_name in not_usable_videos:
            true_positives += 1
        elif not predicted_not_usable and video_name in not_usable_videos:
            false_negatives += 1

    if (true_positives + false_negatives) == 0:
        return None

    return true_positives / (true_positives + false_negatives)

def calculate_precision(predictions, labels):
    """Calcule la précision du filtre par rapport aux étiquettes NotUsable des biologistes.

    Répond à la question : le filtre rejette-t-il des vidéos que les biologistes jugeaient utilisables ?
    Précision = vrais positifs / (vrais positifs + faux positifs).

    Proche de 1 : presque toutes les vidéos rejetées par le filtre étaient NotUsable.
    Proche de 0 : le filtre rejette surtout des vidéos utilisables.
    (à lire avec le rappel : ne rejeter qu'une seule vidéo, à juste titre, donne aussi 1)

    Les vidéos rejetées par le filtre mais absentes de labels sont ignorées.

    Args:
        predictions (dict[str, bool]): Résultat de predict_not_usable.
        labels (dict[str, bool]): Résultat de read_not_usable_labels.

    Returns:
        float | None: Précision entre 0 et 1, ou None si le filtre ne rejette aucune vidéo étiquetée.
    """

    # Vidéos jugées NotUsable par les biologistes
    not_usable_videos = {video_name for video_name, is_not_usable in labels.items() if is_not_usable}

    # Vidéos rejetées par le filtre
    rejected_videos = {video_name for video_name, predicted_not_usable in predictions.items() if predicted_not_usable}

    # Vrais positifs : rejetées par le filtre et par les biologistes
    # Faux positifs : rejetées par le filtre, mais utilisables selon les biologistes
    true_positives = 0
    false_positives = 0

    for video_name in rejected_videos:
        if video_name in not_usable_videos:
            true_positives += 1
        elif video_name in labels:
            false_positives += 1

    if (true_positives + false_positives) == 0:
        return None

    return true_positives / (true_positives + false_positives)