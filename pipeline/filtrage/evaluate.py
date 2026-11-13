import csv
from pathlib import Path

from pipeline.filtrage.frame_io import group_frames_by_video, read_frame_results_csv
from pipeline.filtrage import config
from pipeline.filtrage.models import VideoReport, Cause

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

def find_compared_videos(predictions, labels):
    """Trouve les vidéos comparées : celles qui ont à la fois une prédiction du filtre et une étiquette des biologistes.

    Ce sont les seules vidéos qui comptent dans le rappel et la précision.

    Args:
        predictions (dict[str, bool]): Résultat de predict_not_usable.
        labels (dict[str, bool]): Résultat de read_not_usable_labels.

    Returns:
        set[str]: Noms des vidéos présentes dans predictions et dans labels.
    """

    return predictions.keys() & labels.keys()

def build_video_reports(frame_results):
    """Construit le bilan du filtrage de chaque vidéo : pourcentage de frames rejetées et répartition par cause.

    La vidéo d'une frame est le dossier qui la contient (ex. ".../C00217_2021_03_12_193000_winter_20/f00012.jpg").
    Les pourcentages sont calculés sur le nombre total de frames de la vidéo, donc la somme
    des pourcentages par cause est égale au pourcentage de frames rejetées.

    Args:
        frame_results (list[FrameResult]): Résultats du filtrage des frames.

    Returns:
        list[VideoReport]: Un bilan par vidéo, dans l'ordre de group_frames_by_video.
    """

    video_reports = []
    
    for video_name, video_frames in group_frames_by_video(frame_results).items():
        frame_count = len(video_frames)
        rejected_count = sum(not frame_result.result for frame_result in video_frames)
        rejection_percent = 100 * rejected_count / frame_count

        # Nombre de frames rejetées pour chaque cause (une entrée par membre de Cause, même à 0)
        cause_counts = {cause: 0 for cause in Cause}
        for frame_result in video_frames:
            # Une frame retenue n'a pas de cause (None) : elle n'est comptée dans aucune cause
            if frame_result.cause is not None:
                cause_counts[frame_result.cause] += 1

        cause_percents = {cause: 100 * count / frame_count for cause, count in cause_counts.items()}

        video_reports.append(VideoReport(
            video_name=video_name,
            frame_count=frame_count,
            rejected_percent=rejection_percent,
            cause_percents=cause_percents,
        ))

    return video_reports

# Premières colonnes du CSV du rapport par vidéo, dans l'ordre d'écriture.
# Suivent une colonne par cause (pct_<cause>), puis prediction_not_usable et label_biologistes
CSV_COLUMNS = ("video", "nb_frames", "pct_rejetees")

def export_video_csv(video_reports, predictions, labels, csv_path=config.PATH_TO_VIDEO_REPORT_CSV):
    """Exporte le rapport par vidéo vers un fichier CSV, avec la prédiction du filtre et l'étiquette des biologistes.

    Le CSV contient une ligne par vidéo avec les colonnes :
        - video : nom de la vidéo.
        - nb_frames : nombre de frames de la vidéo.
        - pct_rejetees : pourcentage de frames rejetées (entre 0 et 100).
        - pct_<cause> : une colonne par membre de Cause (ex. pct_brulee), pourcentage de frames
          rejetées pour cette cause. Leur somme est égale à pct_rejetees.
        - prediction_not_usable : True si le filtre prédit la vidéo NotUsable, False sinon.
        - label_biologistes : True si les biologistes l'ont jugée NotUsable, False sinon,
          vide si la vidéo n'est pas étiquetée.

    Args:
        video_reports (list[VideoReport]): Résultat de build_video_reports.
        predictions (dict[str, bool]): Résultat de predict_not_usable.
        labels (dict[str, bool]): Résultat de read_not_usable_labels.
        csv_path (str or Path): Chemin du fichier CSV de sortie (par défaut : PATH_TO_VIDEO_REPORT_CSV).

    Returns:
        list[VideoReport]: Rapports écrits dans le CSV.
    """

    csv_path = Path(csv_path)

    parent_dir = csv_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    header = list(CSV_COLUMNS) + [f"pct_{cause}" for cause in Cause] + ["prediction_not_usable", "label_biologistes"]

    with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)

        for video_report in video_reports:

            csv_writer.writerow([
                video_report.video_name,
                video_report.frame_count,
                round(video_report.rejected_percent, 2),
                *[round(video_report.cause_percents[cause], 2) for cause in Cause],
                predictions.get(video_report.video_name),
                labels.get(video_report.video_name)
            ])

    return video_reports

def evaluate_filter(
        frames_csv_path=config.PATH_TO_CSV, 
        frames_folder=config.PATH_TO_FRAMES, 
        labels_path=config.PATH_TO_LABELS_CSV, 
        report_path=config.PATH_TO_VIDEO_REPORT_CSV):
    """Évalue le filtre par vidéo en le comparant aux étiquettes NotUsable des biologistes.

    Lit le CSV des frames écrit par filter_frames (le filtre n'est pas relancé), prédit
    quelles vidéos sont NotUsable, exporte le rapport par vidéo, puis affiche le nombre
    de vidéos comparées, le rappel et la précision.

    Args:
        frames_csv_path (str or Path): CSV des frames écrit par filter_frames (par défaut : PATH_TO_CSV).
        frames_folder (str or Path): Dossier racine des frames (par défaut : PATH_TO_FRAMES).
        labels_path (str or Path): CSV d'annotation des biologistes (par défaut : PATH_TO_LABELS_CSV).
        report_path (str or Path): CSV du rapport par vidéo à écrire (par défaut : PATH_TO_VIDEO_REPORT_CSV).

    Returns:
        dict: Clés "recall" et "precision" (float entre 0 et 1, ou None si non calculable)
            et "compared_videos" (set[str], vidéos qui ont une prédiction et une étiquette).

    Raises:
        FileNotFoundError: Si le CSV des frames ou celui des biologistes n'existe pas.
        ValueError: Si le CSV des frames vient d'une ancienne version du filtre (colonnes manquantes).
    """

    frame_results_from_csv = read_frame_results_csv(frames_csv_path, frames_folder)
    rejection_percents = video_rejection_percents(frame_results_from_csv)
    predictions = predict_not_usable(rejection_percents)

    labels = read_not_usable_labels(labels_path)

    video_reports = build_video_reports(frame_results_from_csv)
    export_video_csv(video_reports, predictions, labels, report_path)

    recall = calculate_recall(predictions, labels)
    precision = calculate_precision(predictions, labels)
    compared_videos = find_compared_videos(predictions, labels)

    # Vidéos étiquetées par les biologistes, mais sans prédiction (ex. frames non extraites)
    labels_without_prediction = len(labels.keys() - predictions.keys())

    print(f"Rapport par vidéo exporté : {report_path}")
    print(f"Vidéos comparées : {len(compared_videos)} ({labels_without_prediction} étiquetées sans prédiction)")
    print(f"Rappel : {f'{recall:.0%}' if recall is not None else 'non calculable'}")
    print(f"Précision : {f'{precision:.0%}' if precision is not None else 'non calculable'}")

    return {
        "recall": recall,
        "precision": precision,
        "compared_videos": compared_videos,
    }