import pandas as pd
from nlp_features import extract_features
from model import InterviewModel
import numpy as np

# Dataset ejemplo: texto + etiqueta (1 = pasa, 0 = no pasa)
data = pd.read_csv("data.csv")

X = []
y = data["label"].values

for text in data["respuesta"]:
    X.append(extract_features(text))

X = np.array(X)

model = InterviewModel()
model.train(X, y)
model.save()

print("✅ Modelo entrenado y guardado")
