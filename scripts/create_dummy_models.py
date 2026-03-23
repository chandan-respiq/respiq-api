"""
Generate dummy PLS models for development.

Usage:
    uv run scripts/create_dummy_models.py
"""
import pickle
import numpy as np
import yaml
from datetime import date
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

WL_MIN, WL_MAX, WL_STEP = 350.0, 830.0, 0.3
N_SAMPLES = 50
N_COMPONENTS = 5
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)
wavelengths = np.arange(WL_MIN, WL_MAX + WL_STEP, WL_STEP)
wavelength_cols = [str(round(w, 4)) for w in wavelengths]
n_wavelengths = len(wavelengths)

MOLECULES = [
    {"stem": "cyclohexanone", "molecule": "cyclohexanone", "voc_enum": "CYCLOHEXANONE", "unit": "ppb"},
    {"stem": "2-pentanone",   "molecule": "2-pentanone",   "voc_enum": "PENTANONE_2",   "unit": "ppb"},
    {"stem": "methane",       "molecule": "methane",       "voc_enum": "CH4",            "unit": "ppm"},
    {"stem": "h2",            "molecule": "hydrogen",      "voc_enum": "H2",             "unit": "ppm"},
]

for mol in MOLECULES:
    X = rng.random((N_SAMPLES, n_wavelengths))
    y = rng.uniform(0, 500, N_SAMPLES)

    model = PLSRegression(n_components=N_COMPONENTS)
    model.fit(X, y)

    pkl_path = MODELS_DIR / f"{mol['stem']}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)

    spec = {
        "molecule": mol["molecule"],
        "voc_enum": mol["voc_enum"],
        "model_type": "PLSRegression",
        "n_components": N_COMPONENTS,
        "wavelength_range": [WL_MIN, WL_MAX],
        "wavelength_resolution": WL_STEP,
        "unit": mol["unit"],
        "trained_at": str(date.today()),
        "training_experiment_ids": [],
        "notes": "Dummy model for development — random data, not for production use",
    }
    yaml_path = MODELS_DIR / f"{mol['stem']}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(spec, f, sort_keys=False)

    print(f"Created {pkl_path.name} + {yaml_path.name}  ({n_wavelengths} wavelengths)")

print("Done.")
