import numpy as np
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

    TODO: replace NotImplementedError with the respiq preprocessing call, e.g.:
        from respiq.workflows import preprocess_raw
        df = pd.read_csv(io.BytesIO(csv_bytes))
        X = preprocess_raw(df, bundle["wavelength_cols"])
        concentration = float(bundle["model"].predict(X).ravel()[0])
        return PredictionResponse(
            molecule=molecule,
            concentration=round(concentration, 4),
            unit=bundle["spec"]["unit"],
        )
    """
    raise NotImplementedError("Raw preprocessing not yet implemented.")
