import random
from pathlib import Path

import pandas as pd

# Noms des colonnes du CSV (à changer si le vrai CSV est différent)
COL_VIDEO = "video"
COL_COLLIER = "collier"
COL_SAISON = "saison"

CHEMIN_CSV = "data/faux_inventaire.csv"
DOSSIER_SORTIE = "data/decoupage"


def lire_colliers(chemin_csv):
    table = pd.read_csv(chemin_csv)
    colliers = sorted(table[COL_COLLIER].unique())
    return table, colliers


def repartir_colliers(colliers, graine=42):
    # On mélange les colliers, avec un hasard fixé (graine)
    melange = list(colliers)
    random.Random(graine).shuffle(melange)
    train = melange[:5]
    val = melange[5:6]
    test = melange[6:]
    return train, val, test


def verifier_aucun_collier_commun(train, val, test):
    # Arrête le programme si un collier est dans deux groupes
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise ValueError("Un collier apparait dans deux groupes !")


def videos_du_groupe(table, colliers_du_groupe):
    # Garde seulement les lignes dont le collier est dans le groupe
    return table[table[COL_COLLIER].isin(colliers_du_groupe)]


def ecrire_groupes(table, train, val, test):
    dossier = Path(DOSSIER_SORTIE)
    dossier.mkdir(parents=True, exist_ok=True)
    groupes = {"train": train, "val": val, "test": test}
    for nom, colliers_du_groupe in groupes.items():
        videos = videos_du_groupe(table, colliers_du_groupe)
        videos.to_csv(dossier / (nom + ".csv"), index=False)
        print(nom, ":", len(videos), "videos")

def verifier_saisons(table, train, val, test):
    groupes = {"train": train, "val": val, "test": test}
    for nom, colliers_du_groupe in groupes.items():
        videos = videos_du_groupe(table, colliers_du_groupe)
        saisons = sorted(videos[COL_SAISON].unique())
        print(nom, "saisons :", saisons)
        if len(saisons) < 4:
            print("  Attention : il manque des saisons dans", nom)


if __name__ == "__main__":
    table, colliers = lire_colliers(CHEMIN_CSV)
    train, val, test = repartir_colliers(colliers)
    verifier_aucun_collier_commun(train, val, test)
    ecrire_groupes(table, train, val, test)
    verifier_saisons(table, train, val, test)