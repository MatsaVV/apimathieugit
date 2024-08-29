from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib



#Créer une instance de FastAPI
app = FastAPI()

# Définir un modèle de requête
class IrisRequest(BaseModel):
    sepal_width: float
    petal_length: float
    petal_width: float

# Charger le modèle lors du démarrage de l'application
@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("iris_regressor.pkl")

# Point de terminaison pour faire une prédiction
@app.post("/predict")
def predict(iris: IrisRequest):
    # Convertir la requête en un tableau NumPy
    data = [[iris.sepal_width, iris.petal_length, iris.petal_width]]

    # Faire une prédiction
    prediction = model.predict(data)

    # Retourner la prédiction
    return {"predicted_sepal_length": prediction[0]}
