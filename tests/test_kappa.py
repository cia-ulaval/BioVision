import numpy as np
import pandas as pd

from annotation.kappa import kappa_par_attribut


def ecrire(chemin, images, neige, eau):
    pd.DataFrame({"image": images, "neige": neige, "eau": eau}).to_csv(chemin, index=False)


def test_fichiers_identiques(tmp_path):
    images = [f"i{k}" for k in range(200)]
    rng = np.random.default_rng(0)
    neige = rng.integers(0, 3, 200)
    eau = rng.integers(0, 2, 200)
    ecrire(tmp_path / "a.csv", images, neige, eau)
    ecrire(tmp_path / "b.csv", images, neige, eau)
    r = kappa_par_attribut(tmp_path / "a.csv", tmp_path / "b.csv",
                           ["neige", "eau"], ordinaux=["neige"])
    assert r["neige"] == 1.0 and r["eau"] == 1.0


def test_etiquettes_au_hasard_proche_de_zero(tmp_path):
    images = [f"i{k}" for k in range(2000)]
    rng = np.random.default_rng(1)
    ecrire(tmp_path / "a.csv", images, rng.integers(0, 3, 2000), rng.integers(0, 2, 2000))
    ecrire(tmp_path / "b.csv", images, rng.integers(0, 3, 2000), rng.integers(0, 2, 2000))
    r = kappa_par_attribut(tmp_path / "a.csv", tmp_path / "b.csv",
                           ["neige", "eau"], ordinaux=["neige"])
    assert abs(r["neige"]) < 0.1 and abs(r["eau"]) < 0.1


def test_ordre_des_lignes_different(tmp_path):
    images = [f"i{k}" for k in range(100)]
    rng = np.random.default_rng(2)
    neige = rng.integers(0, 3, 100)
    eau = rng.integers(0, 2, 100)
    ecrire(tmp_path / "a.csv", images, neige, eau)
    ordre = rng.permutation(100)
    ecrire(tmp_path / "b.csv", [images[i] for i in ordre], neige[ordre], eau[ordre])
    r = kappa_par_attribut(tmp_path / "a.csv", tmp_path / "b.csv",
                           ["neige", "eau"], ordinaux=["neige"])
    assert r["neige"] == 1.0 and r["eau"] == 1.0