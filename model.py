from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Charger le dataset Iris
iris = load_iris()
X, y = iris.data, iris.target

# Supposons que nous voulons prédire la longueur du sépale (colonne 0)
y = X[:, 0]
X = X[:, 1:]  # On enlève la colonne 0 des features

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle de régression RandomForest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Évaluer le modèle
mse = mean_squared_error(y_test, model.predict(X_test))
print(f"Mean Squared Error: {mse:.2f}")

# Sauvegarder le modèle
joblib.dump(model, "iris_regressor.pkl")
