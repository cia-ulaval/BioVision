
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EXTENSIONS_VIDEO, MO_PAR_GO, SECONDES_PAR_HEURE



def inventorie_corpus(dossier_videos):

    videos = []

    for chemin_video in Path(dossier_videos).rglob("*"):

        if chemin_video.suffix.lower() not in EXTENSIONS_VIDEO:
            continue

        taille_mo = chemin_video.stat().st_size / (1024 ** 2)

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
            "taille_mo": taille_mo
        })

    colonnes = ["nom", "chemin", "duree_secondes", "taille_mb"]
    tableau_videos = pd.DataFrame(videos, columns=colonnes)

    tableau_videos.to_csv("inventaire_corpus.csv", index=False)

    print("Nombre total de vidéos :", len(tableau_videos))
    print("Durée totale (heures) :",
          tableau_videos["duree_secondes"].sum() / SECONDES_PAR_HEURE)
    print("Poids total (GB) :",
          tableau_videos["taille_mo"].sum() / MO_PAR_GO)

    return tableau_videos


if __name__ == "__main__":
# Lancer l'inventaire du corpus
df_videos = inventorie_corpus("data/videos")
