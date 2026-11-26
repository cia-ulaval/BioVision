from pathlib import Path
from pipeline.filtrage.config import FRAMES_PER_SECOND

PATH_TO_ANNOTATIONS = Path("/videos/annotations")
PATH_TO_FRAMES = Path("/frames")

# Lots annotés à extraire (un CSV Timelapse par lot, à côté des vidéos)
LOTS_ANNOTES = [
    "to_analyse/Caribou_rand_Fanny1",
    "to_analyse2",
    "to_analyse3",
    "to_analyse4",
]