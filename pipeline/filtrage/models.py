from pathlib import Path
from dataclasses import dataclass
from enum import StrEnum

class Cause(StrEnum):
    BRULEE = "brulee"
    NOIRE = "noire"
    VOILEE = "voilee"
    VIDEO_COURTE = "video_courte"

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
        cause (Cause | None): Raison du rejet, ou None si la frame est retenue.
    """

    path: Path
    darkness: float
    saturation: float
    sharpness: float
    contrast: float
    brightness: float
    result: bool
    cause: Cause | None

@dataclass
class VideoReport:
    """Bilan du filtrage d'une vidéo : combien de ses frames sont rejetées, et pour quelle cause.

    Les pourcentages sont calculés sur le nombre total de frames de la vidéo :
    la somme des valeurs de cause_percents est donc égale à rejected_percent.

    Attributes:
        video_name (str): Nom de la vidéo (dossier de ses frames, ex. "C00217_2021_03_12_193000_winter_20").
        frame_count (int): Nombre de frames de la vidéo.
        rejected_percent (float): Pourcentage de frames rejetées (entre 0 et 100).
        cause_percents (dict[Cause, float]): Pourcentage de frames rejetées pour chaque cause (entre 0 et 100).
    """

    video_name: str
    frame_count: int
    rejected_percent: float
    cause_percents: dict[Cause, float]