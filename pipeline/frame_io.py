from pathlib import Path
import shutil

def get_frames_from_folder(folder_path, extensions=(".jpg", ".png")):
    """Récupère les frames à partir d'un dossier donné selon les extensions spécifiées.

    Args:
        folder_path (str or Path): Chemin du dossier contenant les frames.
        extensions (tuple or str): Extensions de fichiers à inclure (par défaut : (".jpg", ".png")).

    Returns:
        list: Liste des chemins des frames trouvées dans le dossier.
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
    for element in folder.iterdir():
        if element.is_file() and element.suffix.lower() in extensions:
            frames.append(element)

    return sorted(frames)

def export_filtered_frames(frames, output_folder):
    """Exporte les frames filtrées vers un dossier de sortie spécifié.

    Args:
        frames (list): Liste des chemins des frames à exporter.
        output_folder (str or Path): Chemin du dossier de sortie.

    Returns:
        list: Liste des chemins des frames copiées dans le dossier de sortie.
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    frames_copied = []

    for frame in frames:
        frame_copied = shutil.copy(frame, output_folder / Path(frame).name)
        frames_copied.append(Path(frame_copied))

    return frames_copied