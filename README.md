# RespiQ Prediction API

FastAPI service that loads local VOC prediction models and exposes authenticated inference endpoints.

## Features

- Loads model bundles (`.pkl` + `.yaml`) at startup
- Lists available molecules
- Predicts from:
  - a preprocessed spectral row
  - a raw AVS CSV upload
- Bearer-token auth via `API_KEYS`
- Swagger docs at `/docs`

## Requirements

- Python 3.11+
- Docker + Docker Compose (recommended)

## Quick Start (Docker)

1. Copy env template:

```bash
cp .env.example .env
```

2. Edit `.env` and set at least one API key as JSON array (required by current settings parsing):

```env
API_KEYS=["your-token"]
MODELS_DIR=./models
LOG_LEVEL=INFO
```

3. Ensure model files exist in `models/`:

- `*.pkl` model file
- matching `*.yaml` metadata file

For local development you can generate dummy models:

```bash
uv run python scripts/create_dummy_models.py
```

4. Run:

```bash
docker compose up --build
```

5. Open:

- API docs: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`
- Health: `http://localhost:8080/health`

## Auth

Protected endpoints require:

```http
Authorization: Bearer <your-token>
```

## Endpoints

- `GET /`
- `GET /health`
- `GET /molecules`
- `GET /molecules/{molecule}`
- `GET /molecules/{molecule}/schema`
- `POST /predict/{molecule}`
- `POST /predict/{molecule}/raw`

## Example Requests

List molecules:

```bash
curl http://localhost:8080/molecules
```

Predict from preprocessed row:

```bash
curl -X POST "http://localhost:8080/predict/cyclohexanone" \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{"350.0": 0.023, "350.3": 0.024}'
```

Predict from raw CSV:

```bash
curl -X POST "http://localhost:8080/predict/cyclohexanone/raw" \
  -H "Authorization: Bearer your-token" \
  -F "file=@your_spectrum.csv"
```

## Local Run (without Docker)

```bash
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

Then open `http://localhost:8000/docs`.

## Remote Deploy (Domain + Traefik)

Use `docker-compose-remote.yaml` when deploying behind a domain with TLS.

Required environment variables:

```env
API_DOMAIN=api.yourdomain.com
TRAEFIK_ACME_EMAIL=you@yourdomain.com
```

Run:

```bash
docker compose -f docker-compose-remote.yaml up -d --build
```

Notes:

- DNS A/AAAA record for `API_DOMAIN` must point to your server.
- Ports `80` and `443` must be open publicly for Let's Encrypt HTTP challenge.
