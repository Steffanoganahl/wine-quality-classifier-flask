"""
app.py — Flask
--------------
Estructura de carpetas requerida:
    proyecto/
    ├── app.py
    ├── model/
    │   ├── knn_wine_model.pkl
    │   └── scaler.pkl
    └── templates/
        └── index.html

Instalar dependencias:
    pip install flask joblib numpy pandas scikit-learn
"""

from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# --- Cargar modelo y scaler al iniciar ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model", "knn_wine_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))

FEATURE_NAMES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol"
]

QUALITY_LABELS = {
    0: ("Baja", "🍷", "#c0392b"),
    1: ("Media", "🍷🍷", "#e67e22"),
    2: ("Alta", "🍷🍷🍷", "#27ae60")
}

# Valores de referencia para los placeholders del formulario
FEATURE_DEFAULTS = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.70,
    "citric acid": 0.0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", features=FEATURE_NAMES, defaults=FEATURE_DEFAULTS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [float(request.form.get(f, 0)) for f in FEATURE_NAMES]
        input_df = pd.DataFrame([values], columns=FEATURE_NAMES)
        scaled = scaler.transform(input_df)
        prediction = int(model.predict(scaled)[0])
        proba = model.predict_proba(scaled)[0].tolist()

        label, emoji, color = QUALITY_LABELS[prediction]

        return render_template(
            "index.html",
            features=FEATURE_NAMES,
            defaults=dict(zip(FEATURE_NAMES, values)),
            result={
                "label": label,
                "emoji": emoji,
                "color": color,
                "class": prediction,
                "proba": {
                    "baja": round(proba[0] * 100, 1),
                    "media": round(proba[1] * 100, 1),
                    "alta": round(proba[2] * 100, 1),
                }
            }
        )

    except Exception as e:
        return render_template(
            "index.html",
            features=FEATURE_NAMES,
            defaults=FEATURE_DEFAULTS,
            error=str(e)
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Endpoint JSON — útil para testing con curl o Postman."""
    try:
        data = request.get_json()
        values = [float(data.get(f, 0)) for f in FEATURE_NAMES]
        input_df = pd.DataFrame([values], columns=FEATURE_NAMES)
        scaled = scaler.transform(input_df)
        prediction = int(model.predict(scaled)[0])
        proba = model.predict_proba(scaled)[0].tolist()
        label, emoji, _ = QUALITY_LABELS[prediction]

        return jsonify({
            "prediction": prediction,
            "label": label,
            "emoji": emoji,
            "probabilities": {
                "baja": round(proba[0], 4),
                "media": round(proba[1], 4),
                "alta": round(proba[2], 4)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
