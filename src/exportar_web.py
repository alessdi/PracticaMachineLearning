"""Exporta los modelos entrenados (.keras) a JSON para la página web (docs/).

La página ejecuta las redes en el navegador con JavaScript puro, por lo que aquí se
guardan los pesos de cada capa, los parámetros de estandarización, ejemplos del
conjunto de prueba y las salidas de Keras para verificar que el navegador calcula lo mismo.
Ejecutar desde la carpeta raíz del repositorio:  python src/exportar_web.py
"""
import json, os
import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

SEMILLA = 42
SALIDA = "docs/modelos"
os.makedirs(SALIDA, exist_ok=True)


def redondear(a, d=6):
    return np.round(np.asarray(a, dtype="float64"), d).tolist()


def capas(modelo):
    """Lista de capas con sus pesos en un formato sencillo para JavaScript."""
    lista = []
    for capa in modelo.layers:
        tipo = capa.__class__.__name__
        w = capa.get_weights()
        if tipo == "Dense":
            lista.append({"tipo": "dense", "W": redondear(w[0]), "b": redondear(w[1]),
                          "act": capa.get_config()["activation"]})
        elif tipo == "Conv1D":
            lista.append({"tipo": "conv1d", "W": redondear(w[0]), "b": redondear(w[1]),
                          "act": capa.get_config()["activation"]})
        elif tipo == "BatchNormalization":
            gamma, beta, media, var = w
            escala = gamma / np.sqrt(var + capa.epsilon)          # BN en inferencia: y = x*escala + desplazamiento
            lista.append({"tipo": "bn", "escala": redondear(escala), "desp": redondear(beta - media * escala)})
        elif tipo == "ReLU":
            lista.append({"tipo": "relu"})
        elif tipo == "GlobalAveragePooling1D":
            lista.append({"tipo": "gap"})
    return lista


def guardar(nombre, datos):
    with open(f"{SALIDA}/{nombre}.json", "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, separators=(",", ":"))
    print(f"{nombre}.json  {os.path.getsize(f'{SALIDA}/{nombre}.json')/1024:.0f} KB")


# ---------- 1. Clasificación Dry Bean ----------
df = pd.read_csv("data/Dry_Bean_Dataset.csv")
X = df.drop(columns="Class").values.astype("float32")
cod = LabelEncoder(); y = cod.fit_transform(df["Class"])
X_ent, X_tmp, y_ent, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEMILLA)
X_val, X_pru, y_val, y_pru = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEMILLA)
esc = StandardScaler().fit(X_ent)
modelo = keras.models.load_model("resultados/modelo_drybean.keras")
rng = np.random.default_rng(1)
idx = np.concatenate([rng.choice(np.where(y_pru == c)[0], 6, replace=False) for c in range(len(cod.classes_))])
guardar("clasificacion", {
    "clases": list(cod.classes_), "atributos": list(df.columns[:-1]),
    "media": redondear(esc.mean_), "desv": redondear(esc.scale_),
    "min": redondear(X.min(0), 8), "max": redondear(X.max(0), 8),
    "mediana_clase": {c: redondear(np.median(X[y == k], 0), 8) for k, c in enumerate(cod.classes_)},
    "ejemplos": [{"x": redondear(X_pru[i], 8), "y": int(y_pru[i])} for i in idx],
    "keras": redondear(modelo.predict(esc.transform(X_pru[idx[:5]]), verbose=0)),
    "capas": capas(modelo)})

# ---------- 2. Regresión CCPP ----------
df = pd.read_csv("data/CCPP.csv")
X = df[["AT", "V", "AP", "RH"]].values.astype("float32"); Y = df["PE"].values.astype("float32").reshape(-1, 1)
X_ent, X_tmp, y_ent, y_tmp = train_test_split(X, Y, test_size=0.30, random_state=SEMILLA)
X_val, X_pru, y_val, y_pru = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEMILLA)
esc_x = StandardScaler().fit(X_ent); esc_y = StandardScaler().fit(y_ent)
modelo = keras.models.load_model("resultados/modelo_ccpp.keras")
pred = esc_y.inverse_transform(modelo.predict(esc_x.transform(X_pru), verbose=0)).ravel()
guardar("regresion", {
    "atributos": ["AT", "V", "AP", "RH"],
    "media_x": redondear(esc_x.mean_), "desv_x": redondear(esc_x.scale_),
    "media_y": float(esc_y.mean_[0]), "desv_y": float(esc_y.scale_[0]),
    "min": redondear(X.min(0), 2), "max": redondear(X.max(0), 2),
    "prueba": {"x": redondear(X_pru[:400], 2), "real": redondear(y_pru[:400].ravel(), 2), "pred": redondear(pred[:400], 2)},
    "keras": redondear(pred[:5], 4),
    "capas": capas(modelo)})

# ---------- 3. Series de tiempo FordA ----------
datos = np.loadtxt("data/FordA_TEST.tsv", delimiter="\t")
y_pru = (datos[:, 0] == 1).astype(int); x_pru = datos[:, 1:].astype("float32")
modelo = keras.models.load_model("resultados/modelo_forda.keras")
idx = np.concatenate([rng.choice(np.where(y_pru == c)[0], 20, replace=False) for c in (0, 1)])
rng.shuffle(idx)
guardar("series", {
    "clases": ["Sin falla", "Con falla"],
    "ejemplos": [{"id": int(i), "x": redondear(x_pru[i], 4), "y": int(y_pru[i])} for i in idx],
    "keras": redondear(modelo.predict(x_pru[idx[:5], :, None], verbose=0)),
    "capas": capas(modelo)})
