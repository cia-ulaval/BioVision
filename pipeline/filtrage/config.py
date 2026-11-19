from pathlib import Path

# Une vidéo dont le pourcentage de frames rejetées dépasse ce seuil est prédite NotUsable
VIDEO_REJECT_PERCENT = 50.0

# Récupération des frames en utilisant un chemin d'accès à un dossier contenant les frames
PATH_TO_FRAMES = Path(__file__).parent.parent.parent / "data" / "frames" / "caribou"

# Chemin pour le csv des résultats
PATH_TO_CSV = Path(__file__).parent.parent.parent / "data" / "results" / "resultat_filtrage.csv"

# Chemin du rapport par vidéo : pourcentage de frames rejetées, répartition par cause,
# prédiction du filtre et étiquette des biologistes
PATH_TO_VIDEO_REPORT_CSV = Path(__file__).parent.parent.parent / "data" / "results" / "rapport_videos.csv"

# Chemin du fichier d'annotation des biologistes (colonnes "File" et "NotUsable")
PATH_TO_LABELS_CSV = Path(__file__).parent.parent.parent / "data" / "labels" / "Timelapse_videos_caribou_Fanny2.csv"

# Taille maximale du côté pour le redimensionnement des frames
MAX_SIDE = 640  

# Texture : une variance du Laplacien en dessous de ce seuil est considérée comme un aplat sans texture.
# Ne sert jamais seul (une frame floue reste gardée) : seulement combiné à la saturation ou à la noirceur.
# Valeur provisoire, à calibrer avec la colonne NotUsable des biologistes
TEXTURE_THRESHOLD = 5.0

# Un pixel jusqu'à ce niveau de gris est considéré comme noir (les caméras descendent vers 2-3, pas 0)
DARK_PIXEL_THRESHOLD = 10
# Une frame dont le pourcentage de pixels noirs dépasse ce seuil est considérée comme sous-exposée
DARK_PERCENT_THRESHOLD = 25.0

# Un pixel à partir de ce niveau de gris est considéré comme saturé (les caméras plafonnent vers 246, pas 255)
SATURATED_PIXEL_THRESHOLD = 240
# Une frame dont le pourcentage de pixels saturés dépasse ce seuil est considérée comme surexposée
SATURATED_PERCENT_THRESHOLD = 25.0

# Cadence d'extraction des frames
FRAMES_PER_SECOND = 1
# Une vidéo dont la durée ne dépasse pas ce seuil (en secondes) est rejetée en entier.
MIN_VIDEO_DURATION = 1.0

# Voile (buée, givre) : une frame est voilée si son contraste ET sa texture sont bas ET sa luminosité est moyenne.
# Contraste (écart-type des niveaux de gris) en dessous duquel l'image est un aplat uniforme
VEIL_CONTRAST_THRESHOLD = 12.0
# Texture en dessous de laquelle un aplat peut être un voile.
VEIL_TEXTURE_THRESHOLD = 20.0
# Luminosité (moyenne des niveaux de gris) d'un voile. En dessous : scène de nuit, peu contrastée mais utilisable.
# Au-dessus : frame brûlée, déjà gérée par SATURATED_PERCENT_THRESHOLD
VEIL_BRIGHTNESS_MIN = 60.0
VEIL_BRIGHTNESS_MAX = 230.0