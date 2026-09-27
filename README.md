# Predicción de contratación de plazo fijo — Bank Marketing

**TP Final · Coderhouse — Data Science I** · Autor: Felipe Boivin

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/F-Boivin/tp-final-coderhouse-data-science-1/blob/main/tp-final-coderhouse-data-science-1.ipynb)

Modelo de clasificación con **Random Forest** que predice si un cliente de un banco va a
contratar un depósito a plazo fijo, a partir de sus datos demográficos, financieros y de la
campaña de marketing telefónico. Usa el dataset
[Bank Marketing de UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing): 4.521
clientes y 17 variables.

## Hipótesis

- **H0:** los datos del cliente no alcanzan para predecir la contratación mejor que el azar.
- **H1:** un Random Forest puede predecirla con un accuracy mayor al 75 % y un AUC mayor a 0,5.

La variable `duration` (duración de la llamada) queda afuera: se conoce recién después de
llamar, así que no sirve para decidir a quién llamar.

## Qué hace el notebook

1. **EDA breve:** distribución de la variable objetivo, variables numéricas y categóricas, correlaciones.
2. **Preprocesamiento:** excluye `duration`, codifica el objetivo con `LabelEncoder` y las variables categóricas con One-Hot Encoding (41 features).
3. **Selección de features:** correlación con el objetivo (método de filtro) e importancia según un Random Forest preliminar (método embebido). Quedan las 15 más importantes.
4. **Modelo:** pipeline `StandardScaler` + `RandomForestClassifier` de 100 árboles, con un split 75/25 estratificado.
5. **Evaluación:** accuracy, matriz de confusión, reporte de clasificación, curva ROC con su AUC e importancia de features.
6. **Estabilidad:** reentrena con el 25, 50, 75 y 100 % del set de entrenamiento.
7. **Anexo:** un segundo modelo con `class_weight='balanced'` para atacar el desbalance de clases.

## Resultados

Sobre el set de prueba (1.131 clientes, de los cuales 130 contrataron):

| Métrica | Modelo base | Con `class_weight='balanced'` |
|---|---|---|
| Accuracy | 0,88 | 0,89 |
| AUC-ROC | 0,67 | 0,67 |
| Recall de la clase «Sí» | 0,12 | 0,11 |
| Precisión de la clase «Sí» | 0,44 | 0,56 |

- **H1 se sostiene:** el accuracy supera el 75 % y el AUC supera 0,5.
- **El accuracy engaña:** solo el 11,5 % de los clientes contrató, así que predecir siempre «No» ya acierta el 88 %. El modelo base detecta 15 de los 130 clientes que contrataron.
- **Las variables que más pesan** son `balance`, `age`, `day`, `campaign` y `pdays`.
- **El modelo es estable:** con menos datos de entrenamiento, el accuracy queda entre 0,88 y 0,89 y el AUC entre 0,65 y 0,67.
- **El desbalance es la principal limitación:** ponderar las clases subió la precisión de la clase «Sí», pero no mejoró su recall.

## Cómo correrlo

Lo más simple es el botón **Open in Colab** de arriba. El notebook descarga `bank.csv`
directo de este repositorio, así que no hace falta subir nada.

Para correrlo en tu máquina:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook tp-final-coderhouse-data-science-1.ipynb
```

## Archivos

| Archivo | Contenido |
|---|---|
| `tp-final-coderhouse-data-science-1.ipynb` | El notebook con el análisis completo y sus resultados |
| `bank.csv` | Dataset Bank Marketing de UCI (4.521 filas, separado por `;`) |

## Fuente de los datos

S. Moro, P. Rita y P. Cortez (2014). *Bank Marketing*. UCI Machine Learning Repository.
<https://archive.ics.uci.edu/dataset/222/bank+marketing>
