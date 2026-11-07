from pipeline.filtrage.frame_io import get_frames_from_folder, export_frame_results_csv, group_frames_by_video
from pipeline.filtrage.models import FrameResult
from pipeline.filtrage import config
from pipeline.filtrage.scores import calculate_scores

def validate_frame_quality(frame):
    """Valide la qualité d'une frame à partir de ses scores (calculate_scores).

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
        FileNotFoundError: Si la frame n'existe pas ou ne peut pas être lue (levée par calculate_scores).
    """

    scores = calculate_scores(frame)

    has_texture = scores["sharpness"] >= config.TEXTURE_THRESHOLD
    is_burned = scores["saturation"] > config.SATURATED_PERCENT_THRESHOLD and not has_texture
    is_dark = scores["darkness"] > config.DARK_PERCENT_THRESHOLD and not has_texture
    is_veil = (
        scores["contrast"] < config.VEIL_CONTRAST_THRESHOLD
        and scores["sharpness"] < config.VEIL_TEXTURE_THRESHOLD
        and config.VEIL_BRIGHTNESS_MIN <= scores["brightness"] <= config.VEIL_BRIGHTNESS_MAX
    )

    result = not is_burned and not is_dark and not is_veil

    return FrameResult(path=frame, **scores, result=result)

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
        if len(video_frames) <= config.FRAMES_PER_SECOND * config.MIN_VIDEO_DURATION:
            for frame in video_frames:
                frame.result = False

    return frame_results

def filter_frames(input_folder=config.PATH_TO_FRAMES, csv_path=config.PATH_TO_CSV):
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