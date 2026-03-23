import logging
import pickle
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)
_models: dict = {}


def load_models(models_dir: str):
    p = Path(models_dir)
    for pkl_path in p.glob("*.pkl"):
        yaml_path = pkl_path.with_suffix(".yaml")
        if not yaml_path.exists():
            logger.warning(f"No YAML spec found for {pkl_path.name}, skipping")
            continue

        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        with open(yaml_path) as f:
            spec = yaml.safe_load(f)

        actual_type = type(model).__name__
        if actual_type != spec.get("model_type"):
            logger.warning(
                f"{pkl_path.stem}: model_type in YAML is '{spec.get('model_type')}' "
                f"but loaded type is '{actual_type}'"
            )

        wavelength_cols = _make_wavelength_cols(
            spec["wavelength_range"], spec["wavelength_resolution"]
        )

        _models[pkl_path.stem] = {
            "model": model,
            "wavelength_cols": wavelength_cols,
            "spec": spec,
        }
        logger.info(f"Loaded model: {pkl_path.stem} ({spec['unit']}, {actual_type})")

    if not _models:
        raise RuntimeError(f"No models found in {models_dir}")


def _make_wavelength_cols(wavelength_range: list, resolution: float) -> list[str]:
    wl_min, wl_max = wavelength_range
    wavelengths = np.arange(wl_min, wl_max + resolution, resolution)
    return [str(round(w, 4)) for w in wavelengths]


def get_model(molecule: str) -> dict:
    if molecule not in _models:
        raise KeyError(f"No model for '{molecule}'")
    return _models[molecule]


def list_molecules() -> list[str]:
    return list(_models.keys())
