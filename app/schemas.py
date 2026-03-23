from pydantic import BaseModel


class PredictionResponse(BaseModel):
    molecule: str
    concentration: float
    unit: str
