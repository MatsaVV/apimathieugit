from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from contextlib import asynccontextmanager

# Créer une instance de FastAPI
app = FastAPI()
model = joblib.load("iris_regressor.pkl")

# Définir un modèle de requête
class IrisRequest(BaseModel):
    sepal_width: float
    petal_length: float
    petal_width: float

# Point de terminaison pour faire une prédiction
@app.post("/predict")
def predict(iris: IrisRequest):
    # Convertir la requête en un tableau NumPy
    data = [[iris.sepal_width, iris.petal_length, iris.petal_width]]

    # Faire une prédiction
    prediction = model.predict(data)

    # Retourner la prédiction
    return {"predicted_sepal_length": prediction[0]}

@app.get("/quentin")
def quentin():
    return {"message" : "Je suis un Excel Hero"}
