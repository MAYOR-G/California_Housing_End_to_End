from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError
import numpy as np, joblib
from pathlib import Path

app = FastAPI(title="California House Price API", version="1.0.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

templates = Jinja2Templates(directory="templates")

_MODEL = None
_MODEL_PATH = Path("model.joblib")

def _get_model():
    global _MODEL
    if _MODEL is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError("model.joblib not found. Run the training notebook first.")
        _MODEL = joblib.load(_MODEL_PATH)
    return _MODEL

class HouseInput(BaseModel):
    MedInc: float = Field(..., description="Median income in 10k USD")
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: int
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/predict")
async def api_predict(payload: dict):
    try:
        data = HouseInput(**payload)
        model = _get_model()
        X = np.array([[
            data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms,
            data.Population, data.AveOccup, data.Latitude, data.Longitude
        ]], dtype=float)
        yhat = float(model.predict(X)[0]) * 100_000  # convert from $100k units → USD
        return JSONResponse({
            "prediction_usd": round(yhat, 2),
            "inputs": data.model_dump()
        })
    except ValidationError as ve:
        return JSONResponse({"detail": ve.errors()}, status_code=422)
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)
