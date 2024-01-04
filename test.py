# Importer les bibliothèques nécessaires
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

# Charger le modèle
with open('modele_bcancer.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Charger ou préparer les données de test
url = "https://github.com/AbdallahTayeb/DevOps-Course/raw/main/sample.csv"
test_data = pd.read_csv(url)

# Séparer les fonctionnalités et les étiquettes
X_test = test_data.drop('target', axis=1)
y_test = test_data['target']

# Faire des prédictions
y_pred = model.predict(X_test)

# Calculer la précision
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle sur les nouvelles données : {accuracy * 100:.2f}%")

# Définir un seuil de classification
seuil = 0.5

# Vérifier si le seuil est atteint
predictions_seuil = (model.predict_proba(X_test)[:, 1] > seuil).astype(int)
accuracy_seuil = accuracy_score(y_test, predictions_seuil)
print(f"Précision du modèle avec seuil de classification : {accuracy_seuil * 100:.2f}%")
