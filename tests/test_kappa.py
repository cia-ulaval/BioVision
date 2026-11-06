import pandas as pd
from annotation.kappa import kappa_par_attribut

def test_fichiers_identiques(tmp_path):
    df = pd.DataFrame({"image": [f"i{k}" for k in range(50)],
                       "neige": [k % 3 for k in range(50)],
                       "eau": [k % 2 for k in range(50)]})
    df.to_csv(tmp_path / "a.csv", index=False)
    df.to_csv(tmp_path / "b.csv", index=False)
    r = kappa_par_attribut(tmp_path / "a.csv", tmp_path / "b.csv",
                           ["neige", "eau"], ordinaux=["neige"])
    assert r["neige"] == 1.0 and r["eau"] == 1.0