# Práctica Machine Learning – Redes neuronales (clasificación, regresión y series de tiempo)

**Sistemas Inteligentes – Facultad de Ingeniería, Universidad Autónoma de Campeche**
Alumno: Alessandro Díaz · Profesor: Justino Ramírez Ortegón

Tres redes neuronales en **Python 3 + TensorFlow/Keras** aplicadas a problemas reales con datasets públicos.

## 🔗 Página para probar las 3 redes

**https://alessdi.github.io/PracticaMachineLearning/**

Interfaz web donde se prueban los tres modelos entrenados directamente en el navegador (sin instalar nada):
- **Clasificación:** escribe o carga las 16 medidas de un grano y la red muestra la variedad y la probabilidad de cada clase.
- **Regresión:** mueve los controles de temperatura, vacío, presión y humedad y ve la potencia estimada en MW.
- **Series de tiempo:** elige una serie del sensor (o sube un CSV de 500 valores), agrega ruido y ve el diagnóstico con falla / sin falla.

Los pesos se exportan con `src/exportar_web.py` y la red se calcula en JavaScript (`docs/red.js`), verificado contra Keras.

## Entrenamiento (Google Colab)

El entrenamiento completo de cada red se puede ejecutar en línea con Google Colab:

| # | Tipo de red | Problema real | Dataset | Resultado (prueba) | Ejecutar |
|---|---|---|---|---|---|
| 1 | Pattern recognition (clasificación) | Clasificar 7 variedades de frijol seco por visión artificial (industria agrícola) | [Dry Bean – UCI](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) | Exactitud 92.70 % | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alessdi/PracticaMachineLearning/blob/main/notebooks/01_clasificacion_drybean.ipynb) |
| 2 | Fitting (regresión) | Predecir la potencia eléctrica de una central de ciclo combinado (industria energética) | [Combined Cycle Power Plant – UCI](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant) | R² = 0.948 · RMSE = 3.88 MW | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alessdi/PracticaMachineLearning/blob/main/notebooks/02_regresion_ccpp.ipynb) |
| 3 | Series de tiempo (clasificación) | Detectar fallas en motores a partir de la señal de un sensor (industria automotriz, Ford) | [FordA – UCR Archive](https://www.timeseriesclassification.com/description.php?Dataset=FordA) | Exactitud 93.79 % | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alessdi/PracticaMachineLearning/blob/main/notebooks/03_series_tiempo_forda.ipynb) |

## Cómo ejecutar

### Opción A – En línea (Google Colab, recomendado)
1. Dar clic en el botón **Open in Colab** del ejemplo.
2. Menú **Entorno de ejecución → Ejecutar todas** (Runtime → Run all).
3. Los datos se descargan solos desde la fuente oficial (o desde la carpeta `data/` de este repositorio si la fuente no responde). Las gráficas y métricas aparecen debajo de cada celda.

### Opción B – En la computadora
```bash
git clone https://github.com/alessdi/PracticaMachineLearning.git
cd PracticaMachineLearning
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd notebooks
python ../src/01_clasificacion_drybean.py
python ../src/02_regresion_ccpp.py
python ../src/03_series_tiempo_forda.py
```
Las figuras, tablas de ajuste y modelos entrenados (`.keras`) se guardan en `resultados/`.
También se pueden abrir los notebooks con `jupyter notebook notebooks/`.

## Estructura
```
data/        copias de los datasets (CSV/TSV)
notebooks/   los 3 programas como notebooks ejecutados (con salidas)
src/         los mismos programas como scripts .py (+ exportar_web.py)
docs/        página web para probar las redes (GitHub Pages)
resultados/  figuras, tablas de ajuste de parámetros y modelos entrenados
```

## Referencias de los datos
- Koklu, M. y Ozkan, I. A. (2020). Multiclass classification of dry beans using computer vision and machine learning techniques. *Computers and Electronics in Agriculture*, 174.
- Tüfekci, P. (2014). Prediction of full load electrical power output of a base load operated combined cycle power plant using machine learning methods. *Int. J. of Electrical Power & Energy Systems*, 60.
- Dau, H. A. et al. (2019). The UCR Time Series Archive. *IEEE/CAA Journal of Automatica Sinica*, 6(6).
