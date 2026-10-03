from pathlib import Path

## Récupération des frames en utilisant un chemin d'accès à un dossier contenant les frames
PATH_TO_FRAMES = Path(__file__).parent.parent / "data" / "frames_test" 

# Taille maximale du côté pour le redimensionnement des frames
MAX_SIDE = 640  

## Créer une fonction pour préparer une frame pour le filtrage (par exemple, redimensionner, convertir en niveaux de gris, etc.)
def prepare_frame(frame):
    # 1. Charger l'image en niveaux de gris (cv2.IMREAD_GRAYSCALE)
    # 2. Vérifier que l'image n'est pas None (fichier illisible ou corrompu)
    # 3. Récupérer la hauteur et la largeur de l'image
    # 4. Calculer le facteur d'échelle : MAX_SIDE / plus grand côté
    # 5. Si l'image est plus petite que MAX_SIDE : décider (garder telle quelle ou rejeter)
    # 6. Calculer les nouvelles dimensions (en entiers) avec le facteur d'échelle
    # 7. Redimensionner avec cv2.INTER_AREA
    # 8. Retourner l'image préparée (sert seulement au calcul des scores)
    return None  # Remplacer par le code pour préparer la frame pour le filtrage;

## Flou : variance laplacien en dessous de ce seuil est considéré comme flou
SHARPNESS_THRESHOLD = 100.0

## Exposition : luminosité moyenne en dehors de ce seuil est considérée comme sous-exposée ou surexposée
EXPOSURE_THRESHOLD_LOW = 40.0
EXPOSURE_THRESHOLD_HIGH = 220.0

## Créer une fonction pour calculer le score de netteté d'une frame
def calculate_sharpness_score(frame):
    return None  # Remplacer par le code pour calculer le score de netteté;

## Créer une fonction pour calculer le score d'exposition d'une frame
def calculate_exposure_score(frame):
    return None  # Remplacer par le code pour calculer le score d'exposition;

## Créer une fonction pour calculer le score global d'une frame en combinant les scores de netteté et d'exposition
def calculate_overall_score(sharpness_score, exposure_score):
    return None  # Remplacer par le code pour calculer le score global;

## Créer une fonction pour prendre une décision sur la qualité d'une frame en fonction de ses scores
def decide_frame_quality(sharpness_score, exposure_score):
    return None  # Remplacer par le code pour décider de la qualité de la frame;