from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import joblib

# Init API and rate limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model into memory for inference
classifier = joblib.load("sentiment_classifier.joblib.pkl")

# Classifier endpoint and associated models
class Text(BaseModel):
    text: str = Query(max_length=400)

class Classication(BaseModel):
    positive: bool
    probability: float

@app.post("/classify/", response_model=Classication)
@limiter.limit("20/day")
async def classify(request: Request, text: Text):
    return {
        "positive": int(classifier.predict([text.text])[0]),
        "probability": float(max(classifier.predict_proba([text.text])[0]))
    }