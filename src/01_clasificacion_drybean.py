# %% [markdown]
# # Ejemplo 1 – Red neuronal de reconocimiento de patrones (clasificación)
# **Problema industrial:** clasificación automática de granos de frijol seco en 7 variedades
# a partir de 16 características geométricas obtenidas por visión por computadora
# (sistema de selección de semillas certificadas).
#
# **Dataset:** *Dry Bean Dataset* – UCI Machine Learning Repository (Koklu y Ozkan, 2020),
# 13 611 granos, 16 atributos, 7 clases. https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
#
# **Red:** perceptrón multicapa (MLP) con salida *softmax* y entropía cruzada, Keras/TensorFlow.
#
# Alumno: Alessandro Díaz – Sistemas Inteligentes, UAC.

# %%
import io, os, sys, time, zipfile, urllib.request
import numpy as np
import pandas as pd
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

SEMILLA = 42
keras.utils.set_random_seed(SEMILLA)
RESULTADOS = "../resultados" if os.path.isdir("../data") else "resultados"
os.makedirs(RESULTADOS, exist_ok=True)
print("TensorFlow", tf.__version__)

# %% [markdown]
# ## 1. Carga del dataset (desde la copia local, la fuente oficial UCI o el repositorio)

# %%
URL_UCI = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
URL_REPO = "https://raw.githubusercontent.com/alessdi/PracticaMachineLearning/main/data/Dry_Bean_Dataset.csv"

def cargar_drybean():
    for ruta in ("data/Dry_Bean_Dataset.csv", "../data/Dry_Bean_Dataset.csv"):
        if os.path.exists(ruta):
            return pd.read_csv(ruta), ruta
    try:
        with urllib.request.urlopen(URL_UCI, timeout=60) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        return pd.read_excel(z.open("DryBeanDataset/Dry_Bean_Dataset.xlsx")), URL_UCI
    except Exception as e:
        print("UCI no disponible:", e)
        return pd.read_csv(URL_REPO), URL_REPO

df, origen = cargar_drybean()
print(f"Datos cargados de: {origen}")
print("Forma:", df.shape)
print(df["Class"].value_counts().to_string())
df.head()

# %% [markdown]
# ## 2. Preprocesamiento: división estratificada 70/15/15 y estandarización

# %%
X = df.drop(columns="Class").values.astype("float32")
codificador = LabelEncoder()
y = codificador.fit_transform(df["Class"])
CLASES = list(codificador.classes_)
N_CLASES = len(CLASES)

X_ent, X_tmp, y_ent, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEMILLA)
X_val, X_pru, y_val, y_pru = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEMILLA)

escalador = StandardScaler().fit(X_ent)          # solo con datos de entrenamiento
X_ent, X_val, X_pru = (escalador.transform(a) for a in (X_ent, X_val, X_pru))
print(f"Entrenamiento: {len(X_ent)}  Validación: {len(X_val)}  Prueba: {len(X_pru)}")

# %% [markdown]
# ## 3. Modelo y ajuste de parámetros
# Se comparan varias arquitecturas y tasas de aprendizaje; se elige la de mayor exactitud de validación.

# %%
def crear_modelo(capas, tasa, dropout):
    modelo = keras.Sequential([keras.Input(shape=(X_ent.shape[1],))])
    for n in capas:
        modelo.add(keras.layers.Dense(n, activation="relu"))
        modelo.add(keras.layers.BatchNormalization())
        modelo.add(keras.layers.Dropout(dropout))
    modelo.add(keras.layers.Dense(N_CLASES, activation="softmax"))
    modelo.compile(optimizer=keras.optimizers.Adam(tasa),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo

def entrenar(capas, tasa, dropout, epocas=200, verbose=0):
    keras.utils.set_random_seed(SEMILLA)
    modelo = crear_modelo(capas, tasa, dropout)
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
          keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5)]
    hist = modelo.fit(X_ent, y_ent, validation_data=(X_val, y_val), epochs=epocas,
                      batch_size=64, callbacks=cb, verbose=verbose)
    return modelo, hist

REJILLA = [((32,), 1e-3, 0.1), ((64, 32), 1e-3, 0.2), ((128, 64), 1e-3, 0.2),
           ((128, 64, 32), 1e-3, 0.3), ((64, 32), 3e-3, 0.2)]
resultados = []
for capas, tasa, dropout in REJILLA:
    t0 = time.time()
    modelo, hist = entrenar(capas, tasa, dropout)
    acc_val = modelo.evaluate(X_val, y_val, verbose=0)[1]
    resultados.append({"capas ocultas": str(capas), "tasa": tasa, "dropout": dropout,
                       "épocas": len(hist.history["loss"]), "exactitud val %": round(100 * acc_val, 2),
                       "tiempo s": round(time.time() - t0, 1)})
    print(resultados[-1])
tabla_ajuste = pd.DataFrame(resultados)
tabla_ajuste.to_csv(f"{RESULTADOS}/clasificacion_ajuste.csv", index=False)
print(tabla_ajuste.to_string(index=False))

# %% [markdown]
# ## 4. Entrenamiento final con la mejor configuración

# %%
mejor = tabla_ajuste.loc[tabla_ajuste["exactitud val %"].idxmax()]
capas = tuple(int(c) for c in mejor["capas ocultas"].strip("(),").split(",") if c.strip())
print(f"Mejor configuración: capas={capas}, tasa={mejor['tasa']}, dropout={mejor['dropout']}")
modelo, hist = entrenar(capas, float(mejor["tasa"]), float(mejor["dropout"]), verbose=2)
modelo.summary()
modelo.save(f"{RESULTADOS}/modelo_drybean.keras")

# %% [markdown]
# ## 5. Evaluación en el conjunto de prueba

# %%
y_pred = modelo.predict(X_pru, verbose=0).argmax(axis=1)
acc_ent = accuracy_score(y_ent, modelo.predict(X_ent, verbose=0).argmax(axis=1))
acc_val = accuracy_score(y_val, modelo.predict(X_val, verbose=0).argmax(axis=1))
acc_pru = accuracy_score(y_pru, y_pred)
print(f"Exactitud  entrenamiento: {100*acc_ent:.2f} %  validación: {100*acc_val:.2f} %  prueba: {100*acc_pru:.2f} %")
print(classification_report(y_pru, y_pred, target_names=CLASES, digits=3))

fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
ax[0].plot(hist.history["loss"], label="entrenamiento"); ax[0].plot(hist.history["val_loss"], label="validación")
ax[0].set_title("Entropía cruzada por época"); ax[0].set_xlabel("época"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(hist.history["accuracy"], label="entrenamiento"); ax[1].plot(hist.history["val_accuracy"], label="validación")
ax[1].set_title("Exactitud por época"); ax[1].set_xlabel("época"); ax[1].legend(); ax[1].grid(alpha=.3)
ConfusionMatrixDisplay(confusion_matrix(y_pru, y_pred), display_labels=CLASES).plot(
    ax=ax[2], cmap="Blues", colorbar=False, xticks_rotation=45)
ax[2].set_title(f"Matriz de confusión – prueba ({100*acc_pru:.2f} %)")
fig.suptitle("Ejemplo 1: clasificación de frijol seco (Dry Bean, UCI)", fontsize=14, weight="bold")
fig.tight_layout(); fig.savefig(f"{RESULTADOS}/clasificacion_drybean.png", dpi=130); plt.show()

# %% [markdown]
# ## 6. Uso del modelo: clasificar granos nuevos

# %%
def clasificar_grano(caracteristicas):
    """caracteristicas: lista con las 16 medidas geométricas en el orden del dataset."""
    x = escalador.transform(np.array(caracteristicas, dtype="float32").reshape(1, -1))
    p = modelo.predict(x, verbose=0)[0]
    return CLASES[p.argmax()], p.max()

muestras = df.drop(columns="Class").sample(5, random_state=7)
for i, fila in muestras.iterrows():
    clase, prob = clasificar_grano(fila.values)
    print(f"Grano {i:5d}: real = {df.loc[i, 'Class']:9s} -> red = {clase:9s} (p = {prob:.3f})")
