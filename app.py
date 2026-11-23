import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Seitendesign
st.title("KI-Entscheidungs-Dashboard")
st.sidebar.header("Modell-Parameter")

# Parameter über UI steuern
n_estimators = st.sidebar.slider("Anzahl Bäume", 10, 500, 100)
max_depth = st.sidebar.slider("Maximale Baumtiefe", 1, 20, 10)
min_samples_leaf = st.sidebar.slider("Mindestanzahl Samples pro Blatt", 1, 20, 5)

# Dummy-Daten generieren
np.random.seed(42)
X = np.random.randn(1000, 20)
y = ((X[:, 0] + X[:, 1] > 1) | (X[:, 2] * X[:, 3] > 0.5)).astype(int)

# Modell trainieren
model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    random_state=42
)
model.fit(X, y)

# Feature Importance visualisieren
st.subheader("Feature Importance: Worauf achtet die KI?")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots()
ax.bar(range(X.shape[1]), importances[indices], align="center")
ax.set_xticks(range(X.shape[1]))
ax.set_xticklabels([f"Feature {i+1}" for i in indices], rotation=90)
ax.set_ylabel("Wichtigkeit")
st.pyplot(fig)

# Live-Vorhersage für neue Datenpunkte
st.subheader("Live-Vorhersage")
feature_0 = st.slider("Wert für Feature 1", -3.0, 3.0, 0.0)
feature_1 = st.slider("Wert für Feature 2", -3.0, 3.0, 0.0)

new_data = np.zeros((1, 20))
new_data[0, 0] = feature_0
new_data[0, 1] = feature_1

prediction = model.predict(new_data)
proba = model.predict_proba(new_data)

st.write(f"Vorhersage der KI Klasse: {prediction[0]}")
st.write(f"Wahrscheinlichkeit Klasse 0: {proba[0][0]:.2f} | Klasse 1: {proba[0][1]:.2f}")
