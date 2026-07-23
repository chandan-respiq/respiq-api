from curses import raw
import io
import wave

import numpy as np
import pandas as pd
from app.schemas import PredictionResponse


def from_row(data: dict[str, float], molecule: str, bundle: dict) -> PredictionResponse:
    """Predict from a single preprocessed (ratio-corrected) spectral row."""
    wavelength_cols = bundle["wavelength_cols"]
    missing = [c for c in wavelength_cols if c not in data]
    if missing:
        raise ValueError(f"Missing {len(missing)} wavelength keys, e.g. {missing[:3]}")

    X = np.array([[data[c] for c in wavelength_cols]])
    concentration = float(bundle["model"].predict(X).ravel()[0])
    return PredictionResponse(
        molecule=molecule,
        concentration=round(concentration, 4),
        unit=bundle["spec"]["unit"],
    )


def from_raw_csv(csv_bytes: bytes, molecule: str, bundle: dict) -> PredictionResponse:
    """
    Predict from a raw AVS CSV file.

    Raw CSV format:
    - ~80 rows, ~2051 columns
    - Columns: Time (ms), TimeAbs, <~2048 float wavelength columns>, Ticks
    - Wavelength range: ~296.6 nm to ~855.9 nm

    Native AVS wavelength columns within the model's configured range are
    averaged across all measurement rows to produce one spectrum.
    """
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as e:
        raise ValueError("Could not parse the uploaded CSV file.") from e
    if df.empty:
        raise ValueError("The uploaded CSV contains no spectral rows.")

    try:
        wavelength_min, wavelength_max = map(
            float, bundle["spec"]["wavelength_range"]
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError("The model contains an invalid wavelength range.") from e

    if (
        not np.all(np.isfinite([wavelength_min, wavelength_max]))
        or wavelength_min >= wavelength_max
    ):
        raise ValueError("The model contains an invalid wavelength range.")

    raw_wavelength_columns: list[tuple[str, float]] = []
    for column in df.columns:
        try:
            wavelength = float(column)
        except (TypeError, ValueError):
            continue
        if wavelength_min <= wavelength <= wavelength_max:
            raw_wavelength_columns.append((column, wavelength))

    if not raw_wavelength_columns:
        raise ValueError(
            f"The uploaded CSV has no wavelength columns in the configured "
            f"range {wavelength_min:g}-{wavelength_max:g}."
        )

    raw_wavelength_columns.sort(key=lambda item: item[1])
    wavelength_columns = [
        column for column, _wavelength in raw_wavelength_columns
    ]
    spectral_values = df[wavelength_columns].apply(pd.to_numeric, errors="coerce")
    mean_spectrum = spectral_values.mean(axis=0).to_numpy(dtype=float)
    if not np.all(np.isfinite(mean_spectrum)):
        raise ValueError("One or more wavelength columns contain no valid readings.")

    X = mean_spectrum[np.newaxis, :]
    concentration = float(bundle["model"].predict(X).ravel()[0])
    return PredictionResponse(
        molecule=molecule,
        concentration=round(concentration, 4),
        unit=bundle["spec"]["unit"],
    )
