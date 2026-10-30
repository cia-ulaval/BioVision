import csv
from pathlib import Path

from pipeline.frame_io import group_frames_by_video

# Une vidéo dont le pourcentage de frames rejetées dépasse ce seuil est prédite NotUsable
VIDEO_REJECT_PERCENT = 50.0

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
            not_usable_by_video[video_name] = row["NotUsable"] == "true"

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

    return {video_name: percent > VIDEO_REJECT_PERCENT for video_name, percent in rejection_percents.items()}
