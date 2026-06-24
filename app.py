from fastapi import FastAPI
from pydantic import BaseModel
from model import InterviewModel
from nlp_features import extract_features

app = FastAPI()

model = InterviewModel()
model.load()

class InterviewRequest(BaseModel):
    respuesta: str

@app.get("/")
def root():
    return {"message": "API de evaluación de entrevistas activa"}

@app.post("/evaluar")
def evaluar(req: InterviewRequest):
    # Aquí podrías integrar STT (voz → texto)
    text = req.respuesta
    
    features = extract_features(text)
    decision, score = model.predict(features)
    
    return {
        "decision": decision,
        "score": float(score)
    }