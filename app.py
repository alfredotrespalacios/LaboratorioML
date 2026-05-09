import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ------------------------------------------------------
# Configuración general de la página
# ------------------------------------------------------
st.set_page_config(
    page_title="Laboratorio ML para Finanzas",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Laboratorio Visual de Machine Learning para Finanzas")
st.write(
    "Esta aplicación permite entender una regresión lineal sin escribir código. "
    "El estudiante selecciona variables, entrena el modelo y observa los resultados."
)


# ------------------------------------------------------
# Datos de ejemplo
# ------------------------------------------------------
@st.cache_data
def crear_datos_ejemplo():
    np.random.seed(42)
    n = 80

    inflacion = np.random.uniform(3, 14, n)
    tasa_interes = np.random.uniform(5, 18, n)
    desempleo = np.random.uniform(7, 16, n)
    trm = np.random.uniform(3600, 4800, n)

    # Variable objetivo simulada: crecimiento del PIB
    crecimiento_pib = (
        6
        - 0.18 * inflacion
        - 0.12 * tasa_interes
        - 0.10 * desempleo
        + 0.0004 * (trm - 4000)
        + np.random.normal(0, 0.45, n)
    )

    df = pd.DataFrame({
        "Inflación (%)": inflacion.round(2),
        "Tasa de interés (%)": tasa_interes.round(2),
        "Desempleo (%)": desempleo.round(2),
        "TRM": trm.round(0),
        "Crecimiento PIB (%)": crecimiento_pib.round(2)
    })

    return df


df_ejemplo = crear_datos_ejemplo()


# ------------------------------------------------------
# Carga de archivo o uso de datos de ejemplo
# ------------------------------------------------------
st.sidebar.header("1. Datos")
opcion_datos = st.sidebar.radio(
    "¿Qué datos quieres usar?",
    ["Usar datos de ejemplo", "Subir archivo Excel"]
)

if opcion_datos == "Subir archivo Excel":
    archivo = st.sidebar.file_uploader("Sube un archivo .xlsx", type=["xlsx"])

    if archivo is not None:
        df = pd.read_excel(archivo)
    else:
        st.info("Sube un archivo Excel o selecciona los datos de ejemplo.")
        st.stop()
else:
    df = df_ejemplo

st.subheader("Base de datos")
st.dataframe(df, use_container_width=True)


# ------------------------------------------------------
# Selección de variables
# ------------------------------------------------------
st.sidebar.header("2. Selección del modelo")

columnas_numericas = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

variable_objetivo = st.sidebar.selectbox(
    "Variable objetivo que quieres predecir",
    columnas_numericas,
    index=columnas_numericas.index("Crecimiento PIB (%)") if "Crecimiento PIB (%)" in columnas_numericas else 0
)

variables_explicativas = st.sidebar.multiselect(
    "Variables explicativas",
    [col for col in columnas_numericas if col != variable_objetivo],
    default=[col for col in columnas_numericas if col != variable_objetivo][:3]
)

porcentaje_prueba = st.sidebar.slider(
    "Porcentaje de datos para prueba",
    min_value=10,
    max_value=40,
    value=25,
    step=5
)


# ------------------------------------------------------
# Entrenamiento del modelo
# ------------------------------------------------------
st.sidebar.header("3. Entrenamiento")
entrenar = st.sidebar.button("Entrenar modelo")

if entrenar:
    if len(variables_explicativas) == 0:
        st.error("Selecciona al menos una variable explicativa.")
        st.stop()

    X = df[variables_explicativas]
    y = df[variable_objetivo]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=porcentaje_prueba / 100,
        random_state=42
    )

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Resultados del modelo")

    col1, col2 = st.columns(2)
    col1.metric("Error absoluto medio", f"{mae:.2f}")
    col2.metric("R²", f"{r2:.2f}")

    st.write("### Interpretación sencilla")

    if r2 >= 0.7:
        calidad = "alto"
    elif r2 >= 0.4:
        calidad = "moderado"
    else:
        calidad = "bajo"

    st.write(
        f"El modelo tiene un poder explicativo **{calidad}**. "
        f"El R² de {r2:.2f} indica qué proporción de la variación de "
        f"**{variable_objetivo}** logra explicar el modelo con las variables seleccionadas."
    )

    st.write(
        f"El error absoluto medio es de **{mae:.2f} unidades**. "
        "Esto significa que, en promedio, la predicción del modelo se equivoca aproximadamente "
        f"en {mae:.2f} puntos de la variable objetivo."
    )

    # ------------------------------------------------------
    # Tabla de coeficientes
    # ------------------------------------------------------
    coeficientes = pd.DataFrame({
        "Variable": variables_explicativas,
        "Coeficiente": modelo.coef_.round(4)
    })

    st.write("### Coeficientes del modelo")
    st.dataframe(coeficientes, use_container_width=True)

    st.write(
        "Un coeficiente positivo indica que, cuando esa variable aumenta, la predicción tiende a subir. "
        "Un coeficiente negativo indica que, cuando esa variable aumenta, la predicción tiende a bajar."
    )

    # ------------------------------------------------------
    # Gráfico: valores reales vs predichos
    # ------------------------------------------------------
    st.write("### Valores reales vs. valores predichos")

    resultados = pd.DataFrame({
        "Valor real": y_test.values,
        "Valor predicho": y_pred.round(2)
    })

    fig, ax = plt.subplots()
    ax.scatter(resultados["Valor real"], resultados["Valor predicho"])
    ax.set_xlabel("Valor real")
    ax.set_ylabel("Valor predicho")
    ax.set_title("Comparación entre valores reales y predichos")

    minimo = min(resultados["Valor real"].min(), resultados["Valor predicho"].min())
    maximo = max(resultados["Valor real"].max(), resultados["Valor predicho"].max())
    ax.plot([minimo, maximo], [minimo, maximo])

    st.pyplot(fig)

    st.write("### Tabla de resultados")
    st.dataframe(resultados, use_container_width=True)

else:
    st.info("Selecciona las variables en la barra lateral y presiona **Entrenar modelo**.")
