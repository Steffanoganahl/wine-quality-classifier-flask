# 🍷 Wine Quality Classifier — Flask

App web para clasificación de calidad de vino tinto usando K-Nearest Neighbors (KNN), desplegada con Flask.

**Accuracy del modelo: 84%** — entrenado con Red Wine Quality Dataset (UCI, 1.599 muestras).

---

## Estructura del proyecto

```
wine-flask/
├── app.py
├── requirements_flask.txt
├── Copia_de_Proyecto_K_vecinos_más_Cercanos.ipynb
├── model/
│   ├── knn_wine_model.pkl
│   └── scaler.pkl
└── templates/
    └── index.html
```

---

## Instalación local

```bash
# 1. Crear entorno virtual
/usr/local/opt/python@3.12/bin/python3.12 -m venv venv

# 2. Activar entorno
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements_flask.txt

# 4. Correr la app
python app.py
```

Abrir en: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Despliegue en Render

1. Subir este repositorio a GitHub (incluyendo la carpeta `model/`)
2. En [render.com](https://render.com): **New → Web Service → conectar repo**
3. Configurar:
   - **Build Command:** `pip install -r requirements_flask.txt`
   - **Start Command:** `gunicorn app:app`
   - **Plan:** Free
4. Click **Deploy Web Service**

---

## Funcionamiento

El usuario ingresa los 11 parámetros fisicoquímicos del vino y la app devuelve:
- Clasificación de calidad: **Baja / Media / Alta**
- Probabilidad por cada clase

## Features del modelo

| Feature | Descripción |
|---|---|
| fixed acidity | Acidez fija — tartárico (g/L) |
| volatile acidity | Acidez volátil — acético (g/L) |
| citric acid | Ácido cítrico (g/L) |
| residual sugar | Azúcar residual (g/L) |
| chlorides | Cloruros — sal (g/L) |
| free sulfur dioxide | SO₂ libre (mg/L) |
| total sulfur dioxide | SO₂ total (mg/L) |
| density | Densidad (g/mL) |
| pH | pH |
| sulphates | Sulfatos (g/L) |
| alcohol | Alcohol (% vol.) |

## Clases predichas

| Clase | Etiqueta | Quality original |
|---|---|---|
| 0 | Baja 🍷 | ≤ 4 |
| 1 | Media 🍷🍷 | 5 – 6 |
| 2 | Alta 🍷🍷🍷 | ≥ 7 |

---

## Tech stack

- Python 3.12
- scikit-learn (KNN, GridSearchCV, StandardScaler)
- Flask 3.x
- joblib
- gunicorn (producción)

---

*Steffano · 4Geeks Academy DS & ML Bootcamp*
