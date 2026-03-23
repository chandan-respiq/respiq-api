import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Path, UploadFile

from app.auth import verify_token
from app.config import settings
from app.loader import get_model, list_molecules, load_models
from app.predict import from_raw_csv, from_row
from app.schemas import PredictionResponse

logging.basicConfig(level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models(settings.models_dir)
    yield


app = FastAPI(title="RespiQ Prediction API", version="0.4.0", lifespan=lifespan)


@app.get("/")
def root():
    return {
        "name": "RespiQ Prediction API",
        "version": app.version,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/molecules")
def molecules():
    return {"molecules": list_molecules()}


@app.get("/molecules/{molecule}")
def molecule_details(molecule: str = Path(...)):
    try:
        bundle = get_model(molecule)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    spec = bundle["spec"]
    return {
        "molecule": molecule,
        "model_type": spec.get("model_type"),
        "unit": spec.get("unit"),
        "n_components": spec.get("n_components"),
        "wavelength_range": spec.get("wavelength_range"),
        "wavelength_resolution": spec.get("wavelength_resolution"),
        "trained_at": spec.get("trained_at"),
    }


@app.get("/molecules/{molecule}/schema")
def molecule_input_schema(molecule: str = Path(...)):
    try:
        bundle = get_model(molecule)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    wavelength_cols = bundle["wavelength_cols"]
    return {
        "molecule": molecule,
        "required_key_count": len(wavelength_cols),
        "required_keys": wavelength_cols,
    }


@app.post("/predict/{molecule}", response_model=PredictionResponse)
async def predict_preprocessed(
    molecule: str = Path(...),
    data: dict[str, float] = ...,
    _token: str = Depends(verify_token),
):
    """
    Predict from a single preprocessed spectral row.

        curl -X POST http://localhost:8080/predict/cyclohexanone
          -H "Authorization: Bearer <token>"
          -H "Content-Type: application/json"
          -d '{"350.0": 0.023, "350.3": 0.024, ...}'
    """
    try:
        bundle = get_model(molecule)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        return from_row(data, molecule, bundle)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict/{molecule}/raw", response_model=PredictionResponse)
async def predict_raw(
    molecule: str = Path(...),
    file: UploadFile = File(...),
    _token: str = Depends(verify_token),
):
    """
    Predict from a raw AVS CSV file.

        curl -X POST http://localhost:8080/predict/cyclohexanone/raw
          -H "Authorization: Bearer <token>"
          -F "file=@your_spectrum.csv"
    """
    try:
        bundle = get_model(molecule)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        return from_raw_csv(await file.read(), molecule, bundle)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
