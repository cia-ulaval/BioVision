
from pathlib import Path
import cv2
import pandas as pd


def inventorie_corpus(dossier_videos):

    videos = []

    extensions_video = [".mp4", ".avi", ".mov", ".mkv"]

    for chemin_video in Path(dossier_videos).rglob("*"):

        if chemin_video.suffix.lower() not in extensions_video:
            continue

        taille_MB = chemin_video.stat().st_size / (1024 ** 2)

        capture = cv2.VideoCapture(str(chemin_video))

        fps = capture.get(cv2.CAP_PROP_FPS) # nb frames par seconde
        nombre_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)

        if capture.isOpened() and fps > 0:
            duree_secondes = nombre_frames / fps
        else:
            duree_secondes = None

        capture.release()

        videos.append({
            "nom": chemin_video.name,
            "chemin": str(chemin_video),
            "duree_secondes": duree_secondes,
            "taille_MB": taille_MB
        })

    colonnes = ["nom", "chemin", "duree_secondes", "taille_mb"]
    tableau_videos = pd.DataFrame(videos, columns=colonnes)

    tableau_videos.to_csv("inventaire_corpus.csv", index=False)

    print("Nombre total de vidéos :", len(tableau_videos))
    print("Durée totale (heures) :",
          tableau_videos["duree_secondes"].sum() / 3600)
    print("Poids total (GB) :",
          tableau_videos["taille_mb"].sum() / 1024)

    return tableau_videos


# Lancer l'inventaire du corpus
df_videos = inventorie_corpus("data/videos")
