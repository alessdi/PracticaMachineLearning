# %% [markdown]
# # Ejemplo 3 – Red neuronal para series de tiempo (clasificación de series)
# **Problema industrial:** detectar si un motor automotriz tiene una falla a partir de la serie de
# tiempo de 500 muestras de su sensor de ruido/vibración (diagnóstico de fin de línea, Ford).
#
# **Dataset:** *FordA* – UCR/UEA Time Series Classification Archive (WCCI 2008 Ford Classification
# Challenge). 3 601 series de entrenamiento y 1 320 de prueba, longitud 500, 2 clases.
# https://www.timeseriesclassification.com/description.php?Dataset=FordA
#
# **Red:** red convolucional 1D (FCN, Wang et al. 2017) con 3 bloques Conv1D + BatchNorm + ReLU,
# Global Average Pooling y salida softmax, Keras/TensorFlow.
#
# Alumno: Alessandro Díaz – Sistemas Inteligentes, UAC.

# %%
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

SEMILLA = 42
keras.utils.set_random_seed(SEMILLA)
RESULTADOS = "../resultados" if os.path.isdir("../data") else "resultados"
os.makedirs(RESULTADOS, exist_ok=True)
print("TensorFlow", tf.__version__)

# %% [markdown]
# ## 1. Carga del dataset

# %%
URL_UCR = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
URL_REPO = "https://raw.githubusercontent.com/alessdi/PracticaMachineLearning/main/data/"

def cargar_forda(nombre):
    for ruta in (f"data/{nombre}", f"../data/{nombre}"):
        if os.path.exists(ruta):
            datos = np.loadtxt(ruta, delimiter="\t"); break
    else:
        try:
            datos = np.loadtxt(URL_UCR + nombre, delimiter="\t")
        except Exception as e:
            print("Fuente UCR no disponible:", e)
            datos = np.loadtxt(URL_REPO + nombre, delimiter="\t")
    y = (datos[:, 0] == 1).astype(int)          # -1 -> 0 (sin falla), 1 -> 1 (con falla)
    x = datos[:, 1:].astype("float32")[..., np.newaxis]
    return x, y

x_ent_total, y_ent_total = cargar_forda("FordA_TRAIN.tsv")
x_pru, y_pru = cargar_forda("FordA_TEST.tsv")
CLASES = ["Sin falla (-1)", "Con falla (+1)"]
print("Entrenamiento:", x_ent_total.shape, " Prueba:", x_pru.shape)
print("Balance de clases (entrenamiento):", np.bincount(y_ent_total))

# 15 % del conjunto de entrenamiento se reserva para validación (parada temprana y ajuste)
x_ent, x_val, y_ent, y_val = train_test_split(x_ent_total, y_ent_total, test_size=0.15,
                                              stratify=y_ent_total, random_state=SEMILLA)

fig, ax = plt.subplots(figsize=(12, 3.5))
for c, color in zip([0, 1], ["tab:blue", "tab:red"]):
    for serie in x_ent[y_ent == c][:2]:
        ax.plot(serie.ravel(), color=color, alpha=.8, lw=1)
    ax.plot([], color=color, label=CLASES[c])
ax.set_title("Ejemplos de series del sensor (FordA)"); ax.set_xlabel("muestra"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{RESULTADOS}/series_ejemplos.png", dpi=130); plt.show()

# %% [markdown]
# ## 2. Modelo convolucional 1D y ajuste de parámetros

# %%
def crear_modelo(filtros, kernel, tasa):
    entrada = keras.Input(shape=x_ent.shape[1:])
    h = entrada
    for _ in range(3):
        h = keras.layers.Conv1D(filtros, kernel, padding="same")(h)
        h = keras.layers.BatchNormalization()(h)
        h = keras.layers.ReLU()(h)
    h = keras.layers.GlobalAveragePooling1D()(h)
    salida = keras.layers.Dense(2, activation="softmax")(h)
    modelo = keras.Model(entrada, salida)
    modelo.compile(optimizer=keras.optimizers.Adam(tasa),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo

def entrenar(filtros, kernel, tasa, epocas, verbose=0):
    keras.utils.set_random_seed(SEMILLA)
    modelo = crear_modelo(filtros, kernel, tasa)
    cb = [keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-4),
          keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)]
    hist = modelo.fit(x_ent, y_ent, validation_data=(x_val, y_val), epochs=epocas,
                      batch_size=32, callbacks=cb, verbose=verbose)
    return modelo, hist

# Búsqueda corta (30 épocas) de número de filtros y tamaño de kernel
REJILLA = [(32, 3, 1e-3), (64, 3, 1e-3), (64, 7, 1e-3)]
resultados = []
for filtros, kernel, tasa in REJILLA:
    t0 = time.time()
    modelo, hist = entrenar(filtros, kernel, tasa, epocas=30)
    acc_val = modelo.evaluate(x_val, y_val, verbose=0)[1]
    resultados.append({"filtros": filtros, "kernel": kernel, "tasa": tasa,
                       "exactitud val %": round(100 * acc_val, 2), "tiempo s": round(time.time() - t0, 1)})
    print(resultados[-1])
tabla_ajuste = pd.DataFrame(resultados)
tabla_ajuste.to_csv(f"{RESULTADOS}/series_ajuste.csv", index=False)
print(tabla_ajuste.to_string(index=False))

# %% [markdown]
# ## 3. Entrenamiento final con la mejor configuración

# %%
mejor = tabla_ajuste.loc[tabla_ajuste["exactitud val %"].idxmax()]
print(f"Mejor configuración: filtros={int(mejor['filtros'])}, kernel={int(mejor['kernel'])}, tasa={mejor['tasa']}")
modelo, hist = entrenar(int(mejor["filtros"]), int(mejor["kernel"]), float(mejor["tasa"]), epocas=150, verbose=2)
modelo.summary()
modelo.save(f"{RESULTADOS}/modelo_forda.keras")

# %% [markdown]
# ## 4. Evaluación en el conjunto de prueba oficial (1 320 series nunca vistas)

# %%
prob = modelo.predict(x_pru, verbose=0)
y_pred = prob.argmax(axis=1)
acc_ent = accuracy_score(y_ent, modelo.predict(x_ent, verbose=0).argmax(axis=1))
acc_val = accuracy_score(y_val, modelo.predict(x_val, verbose=0).argmax(axis=1))
acc_pru = accuracy_score(y_pru, y_pred)
print(f"Exactitud  entrenamiento: {100*acc_ent:.2f} %  validación: {100*acc_val:.2f} %  prueba: {100*acc_pru:.2f} %")
print(classification_report(y_pru, y_pred, target_names=CLASES, digits=3))

fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
ax[0].plot(hist.history["loss"], label="entrenamiento"); ax[0].plot(hist.history["val_loss"], label="validación")
ax[0].set_title("Entropía cruzada por época"); ax[0].set_xlabel("época"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(hist.history["accuracy"], label="entrenamiento"); ax[1].plot(hist.history["val_accuracy"], label="validación")
ax[1].set_title("Exactitud por época"); ax[1].set_xlabel("época"); ax[1].legend(); ax[1].grid(alpha=.3)
ConfusionMatrixDisplay(confusion_matrix(y_pru, y_pred), display_labels=CLASES).plot(ax=ax[2], cmap="Blues", colorbar=False)
ax[2].set_title(f"Matriz de confusión – prueba ({100*acc_pru:.2f} %)")
fig.suptitle("Ejemplo 3: detección de fallas en motores con series de tiempo (FordA, UCR)", fontsize=14, weight="bold")
fig.tight_layout(); fig.savefig(f"{RESULTADOS}/series_forda.png", dpi=130); plt.show()

# %% [markdown]
# ## 5. Uso del modelo: diagnosticar series nuevas

# %%
def diagnosticar_motor(serie):
    """serie: arreglo de 500 lecturas normalizadas del sensor."""
    p = modelo.predict(np.asarray(serie, dtype="float32").reshape(1, -1, 1), verbose=0)[0]
    return CLASES[p.argmax()], p.max()

rng = np.random.default_rng(3)
for i in rng.choice(len(x_pru), 6, replace=False):
    clase, p = diagnosticar_motor(x_pru[i])
    print(f"Serie de prueba {i:4d}: real = {CLASES[y_pru[i]]:15s} -> red = {clase:15s} (p = {p:.3f})")
