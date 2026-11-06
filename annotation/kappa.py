import pandas as pd
from sklearn.metrics import cohen_kappa_score


def kappa_par_attribut(fichier_a, fichier_b, attributs, ordinaux=()):
    a = pd.read_csv(fichier_a).set_index("image")
    b = pd.read_csv(fichier_b).set_index("image")
    communes = a.index.intersection(b.index)
    if len(communes) == 0:
        raise ValueError("Aucune image en commun entre les deux fichiers.")

    resultats = {}
    for att in attributs:
        poids = "quadratic" if att in ordinaux else None
        resultats[att] = cohen_kappa_score(
            a.loc[communes, att], b.loc[communes, att], weights=poids
        )
    return resultats