from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from model import InterviewModel
from nlp_features import extract_features
from typing import Dict
import uuid
from pathlib import Path

app = FastAPI()

model = InterviewModel()
model.load()

# -------------------------
# Preguntas de entrevista
# -------------------------
PREGUNTAS = [
    "Cuéntame sobre ti",
    "Describe un desafío laboral y cómo lo resolviste",
    "Cómo manejas conflictos en equipo",
    "Por qué deberíamos contratarte"
]

# -------------------------
# Memoria en RAM (simple)
# -------------------------
sesiones: Dict[str, dict] = {}

# -------------------------
# Modelo request
# -------------------------
class RespuestaRequest(BaseModel):
    session_id: str = None
    respuesta: str = None


@app.get("/", response_class=HTMLResponse)
def read_root():
    html_path = Path(__file__).resolve().parent / "index.html"
    return html_path.read_text(encoding="utf-8")


# -------------------------
# Iniciar entrevista
# -------------------------
@app.post("/start")
def start_interview():
    session_id = str(uuid.uuid4())

    sesiones[session_id] = {
        "indice": 0,
        "respuestas": [],
        "scores": []
    }

    return {
        "session_id": session_id,
        "pregunta": PREGUNTAS[0]
    }


# -------------------------
# Continuar entrevista
# -------------------------
@app.post("/responder")
def responder(req: RespuestaRequest):

    if req.session_id not in sesiones:
        return {"error": "Sesión no válida"}

    sesion = sesiones[req.session_id]

    # Guardar respuesta
    sesion["respuestas"].append(req.respuesta)

    # Evaluar respuesta con ML
    features = extract_features(req.respuesta)
    decision, score = model.predict(features)

    sesion["scores"].append(score)

    # Avanzar pregunta
    sesion["indice"] += 1

    # Si aún hay preguntas
    if sesion["indice"] < len(PREGUNTAS):
        return {
            "pregunta": PREGUNTAS[sesion["indice"]],
            "score_respuesta": float(score)
        }

    # -------------------------
    # Fin de entrevista
    # -------------------------
    score_final = sum(sesion["scores"]) / len(sesion["scores"])

    decision_final = "Avanza" if score_final > 0.7 else "No avanza"

    return {
        "mensaje": "Entrevista finalizada",
        "score_final": float(score_final),
        "decision_final": decision_final,
        "respuestas": sesion["respuestas"]
    }