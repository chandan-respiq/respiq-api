import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.cross_decomposition import PLSRegression

import app.loader as loader_module

DUMMY_MOLECULE = "cyclohexanone"
DUMMY_WAVELENGTH_COLS = ["350.0", "350.3", "350.6", "350.9", "351.2"]
VALID_TOKEN = "test-token-123"

# Build a minimal fitted PLSRegression on 5 wavelengths
_rng = np.random.default_rng(0)
_X = _rng.random((20, len(DUMMY_WAVELENGTH_COLS)))
_y = _rng.uniform(0, 500, 20)
_pls = PLSRegression(n_components=2)
_pls.fit(_X, _y)

DUMMY_BUNDLE = {
    "model": _pls,
    "wavelength_cols": DUMMY_WAVELENGTH_COLS,
    "spec": {
        "unit": "ppb",
        "model_type": "PLSRegression",
        "wavelength_range": [350.0, 351.2],
    },
}


@pytest.fixture(autouse=True)
def inject_models(monkeypatch):
    """Inject dummy model bundle and patch load_models to be a no-op."""
    monkeypatch.setattr(loader_module, "_models", {DUMMY_MOLECULE: DUMMY_BUNDLE})
    monkeypatch.setattr(loader_module, "load_models", lambda _: None)

    # Patch settings so auth works without a real .env
    import app.config as config_module
    monkeypatch.setattr(config_module.settings, "api_keys", [VALID_TOKEN])
    monkeypatch.setattr(config_module.settings, "models_dir", "/tmp/models")


@pytest.fixture
def client(inject_models):
    from app.main import app
    with TestClient(app) as c:
        yield c


def valid_payload():
    return {col: 0.5 for col in DUMMY_WAVELENGTH_COLS}


# 1. Health check
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# 2. Molecules list
def test_molecules(client):
    resp = client.get("/molecules")
    assert resp.status_code == 200
    assert DUMMY_MOLECULE in resp.json()["molecules"]


# 3. Valid prediction
def test_predict_valid(client):
    resp = client.post(
        f"/predict/{DUMMY_MOLECULE}",
        json=valid_payload(),
        headers={"Authorization": f"Bearer {VALID_TOKEN}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["molecule"] == DUMMY_MOLECULE
    assert isinstance(body["concentration"], float)
    assert body["unit"] == "ppb"


# 4. Missing wavelength keys → 422
def test_predict_missing_keys(client):
    resp = client.post(
        f"/predict/{DUMMY_MOLECULE}",
        json={"350.0": 0.5},  # missing the other 4 cols
        headers={"Authorization": f"Bearer {VALID_TOKEN}"},
    )
    assert resp.status_code == 422


# 5. No token → 403 (HTTPBearer returns 403 when header is absent)
def test_predict_no_token(client):
    resp = client.post(f"/predict/{DUMMY_MOLECULE}", json=valid_payload())
    assert resp.status_code == 403


# 6. Wrong token → 401
def test_predict_wrong_token(client):
    resp = client.post(
        f"/predict/{DUMMY_MOLECULE}",
        json=valid_payload(),
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


# 7. Unknown molecule → 404
def test_predict_unknown_molecule(client):
    resp = client.post(
        "/predict/unknown",
        json=valid_payload(),
        headers={"Authorization": f"Bearer {VALID_TOKEN}"},
    )
    assert resp.status_code == 404


# 8. Raw endpoint
def test_predict_raw(client):
    csv_bytes = (
        b"Time,350.0,350.3,350.6,350.9,351.2,Ticks\n"
        b"0,1,2,3,4,5,10\n"
        b"1,3,4,5,6,7,11\n"
    )
    resp = client.post(
        f"/predict/{DUMMY_MOLECULE}/raw",
        files={"file": ("spectrum.csv", csv_bytes, "text/csv")},
        headers={"Authorization": f"Bearer {VALID_TOKEN}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["molecule"] == DUMMY_MOLECULE
    assert isinstance(body["concentration"], float)
    assert body["unit"] == "ppb"


def test_predict_raw_averages_native_wavelengths_in_model_range():
    from app.predict import from_raw_csv

    class RecordingModel:
        def predict(self, X):
            self.X = X
            return np.array([12.34567])

    model = RecordingModel()
    bundle = {
        "model": model,
        "spec": {"unit": "ppb", "wavelength_range": [350.0, 350.7]},
    }
    csv_bytes = (
        b"Time,349.9,350.3,350.7,Ticks\n"
        b"0,1.0,2.0,3.0,10\n"
        b"1,3.0,4.0,5.0,11\n"
    )

    result = from_raw_csv(csv_bytes, DUMMY_MOLECULE, bundle)

    np.testing.assert_allclose(model.X, [[3.0, 4.0]])
    assert result.concentration == 12.3457
