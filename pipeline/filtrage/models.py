from pathlib import Path
from dataclasses import dataclass

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