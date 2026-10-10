from pathlib import Path
import csv

def get_frames_from_folder(folder_path, extensions=(".jpg", ".png")):
    """Récupère les frames d'un dossier et de ses sous-dossiers selon les extensions spécifiées.

    Args:
        folder_path (str or Path): Chemin du dossier contenant les frames.
        extensions (tuple or str): Extensions de fichiers à inclure (par défaut : (".jpg", ".png")).

    Returns:
        list: Liste triée des chemins des frames correspondant aux extensions spécifiées,
            y compris celles des sous-dossiers.

    Raises:
        ValueError: Si folder_path n'existe pas ou n'est pas un dossier.
    """

    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Le chemin spécifié n'existe pas ou n'est pas un dossier : {folder_path}")

    # Vérifier si les extensions sont fournies sous forme de chaîne de caractères ou de tuple/list
    if isinstance(extensions, str):
        extensions = (extensions,)

    # Convertir les extensions en minuscules pour la comparaison et ajouter un point au début si nécessaire
    extensions = tuple(ext.lower() for ext in extensions)
    extensions = tuple(ext if ext.startswith('.') else f".{ext}" for ext in extensions)

    frames = []
    for element in folder.rglob("*"):
        if element.is_file() and element.suffix.lower() in extensions:
            frames.append(element)

    return sorted(frames)

def export_frames_csv(frames, base_folder, csv_path):
    """Exporte les chemins des frames filtrées vers un fichier CSV.

    Le CSV contient une colonne "frame_path" avec le chemin de chaque frame relatif
    à base_folder, toujours écrit avec des "/" (ex. : "video_01/frame_0001.jpg").
    Pour retrouver une frame : base_folder / frame_path.

    Le fichier est créé s'il n'existe pas et remplacé s'il existe déjà.

    Args:
        frames (list): Liste des chemins des frames à exporter.
        base_folder (str or Path): Dossier racine des frames.
        csv_path (str or Path): Chemin du fichier CSV de sortie.

    Returns:
        list: Liste des chemins des frames écrites dans le CSV.

    Raises:
        ValueError: Si une frame ne se trouve pas dans base_folder.
    """

    base_folder = Path(base_folder)
    csv_path = Path(csv_path)

    parent_dir = csv_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['frame_path'])

        for frame in frames:
            relative_path = Path(frame).relative_to(base_folder)
            csv_writer.writerow([relative_path.as_posix()])

    return frames