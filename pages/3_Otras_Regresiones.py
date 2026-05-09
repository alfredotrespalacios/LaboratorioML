from datetime import datetime
import pandas as pd
import streamlit as st

from utils.carga_datos import cargar_datos_modulo, detectar_columnas_numericas
from utils.diagnosticos import metricas_regresion
from utils.exportacion import preparar_excel_generico, preparar_json_descarga
from utils.pdf_reportes import crear_pdf_modulo
from utils.graficos import grafico_histograma_interactivo, grafico_importancia_variables, grafico_real_vs_predicho, grafico_residuales
from utils.modelos import obtener_regresor
from utils.transformaciones import preparar_xy, escalar_train_test
from utils.estilos import mostrar_encabezado, caja_pedagogica

st.set_page_config(page_title="Otras regresiones", page_icon="🌲", layout="wide")
mostrar_encabezado("🌲 Otras regresiones", "Modelos predictivos de regresión con selección de algoritmo.")

caja_pedagogica("Permite comparar algoritmos de regresión no lineal y evaluar el error predictivo.")

df, nombre_archivo, fuente_datos = cargar_datos_modulo("data/datos_otras_regresiones.xlsx", "otras_regresiones")
st.dataframe(df.head(20), use_container_width=True)

num_cols = detectar_columnas_numericas(df)
y_col = st.sidebar.selectbox("Variable objetivo Y", num_cols)
x_cols = st.sidebar.multiselect("Variables explicativas X", [c for c in num_cols if c != y_col], default=[c for c in num_cols if c != y_col][:5])
alg = st.sidebar.selectbox("Algoritmo", ["Árbol de decisión", "Random Forest", "Gradient Boosting", "KNN Regressor", "Support Vector Regression"])
test_size = st.sidebar.slider("Porcentaje de prueba", 10, 40, 25) / 100
escalar = st.sidebar.checkbox("Estandarizar variables X", value=alg in ["KNN Regressor", "Support Vector Regression"])

EXPLICACIONES_REGRESION = {
    "Árbol de decisión": """
    **Árbol de decisión:** divide los datos en reglas sucesivas para predecir la variable objetivo. Es fácil de interpretar,
    pero puede sobreajustarse si el árbol crece demasiado.
    """,
    "Random Forest": """
    **Random Forest:** construye muchos árboles de decisión y combina sus predicciones. Suele ser más estable que un árbol
    individual y puede capturar relaciones no lineales.
    """,
    "Gradient Boosting": """
    **Gradient Boosting:** construye modelos de forma secuencial, donde cada nuevo árbol intenta corregir errores del anterior.
    Puede ser muy preciso, pero requiere cuidado para evitar sobreajuste.
    """,
    "KNN Regressor": """
    **KNN Regressor:** predice usando las observaciones más cercanas. Es intuitivo y depende fuertemente de la escala de las variables,
    por lo que suele requerir estandarización.
    """,
    "Support Vector Regression": """
    **Support Vector Regression:** busca una función que prediga dentro de un margen de tolerancia. Puede capturar relaciones no lineales
    usando kernels, y es sensible a la escala de las variables.
    """
}

with st.expander("Explicación del método seleccionado", expanded=True):
    st.markdown(EXPLICACIONES_REGRESION[alg])

with st.expander("Explicación de configuración: porcentaje de prueba y estandarización", expanded=True):
    st.markdown(
        """
        **Porcentaje de prueba:** proporción de datos que se reserva para evaluar el modelo después del entrenamiento. 
        Por ejemplo, 25% significa que el modelo aprende con el 75% de los datos y se evalúa con el 25% restante.

        **Estandarizar variables X:** transforma las variables explicativas para que tengan escala comparable, usualmente
        media cero y desviación estándar uno. Es especialmente importante en modelos sensibles a la escala, como KNN y
        Support Vector Regression.
        """
    )

if not x_cols:
    st.warning("Seleccione variables explicativas.")
    st.stop()

if "resultados_otras_regresiones_actuales" not in st.session_state:
    st.session_state["resultados_otras_regresiones_actuales"] = None

if st.button("Entrenar modelo"):
    X_train, X_test, y_train, y_test = preparar_xy(df, y_col, x_cols, test_size)
    X_train_model, X_test_model, _ = escalar_train_test(X_train, X_test, escalar)
    modelo = obtener_regresor(alg)
    modelo.fit(X_train_model, y_train)
    y_pred = modelo.predict(X_test_model)
    metricas = metricas_regresion(y_test, y_pred)

    pred_df = pd.DataFrame({"y_real": y_test.values, "y_predicho": y_pred, "error": y_test.values - y_pred})
    importancia_df = pd.DataFrame()
    if hasattr(modelo, "feature_importances_"):
        importancia_df = pd.DataFrame({"variable": x_cols, "importancia": modelo.feature_importances_}).sort_values("importancia")

    st.session_state["resultados_otras_regresiones_actuales"] = {
        "tipo_analisis": "Otras regresiones",
        "fecha_ejecucion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "archivo_usado": nombre_archivo,
        "fuente_datos": fuente_datos,
        "algoritmo": alg,
        "variable_objetivo": y_col,
        "variables_explicativas": x_cols,
        "test_size": test_size,
        "estandarizacion": escalar,
        "metricas": metricas,
        "predicciones": pred_df.to_dict(orient="records"),
        "importancia_variables": importancia_df.to_dict(orient="records"),
    }

res = st.session_state["resultados_otras_regresiones_actuales"]
if res:
    st.subheader("Métricas")
    st.json(res["metricas"])
    st.caption("Estas métricas evalúan el desempeño predictivo fuera de muestra. R² mide capacidad explicativa; MAE y RMSE miden error de predicción.")

    pred_df = pd.DataFrame(res["predicciones"])

    st.plotly_chart(grafico_real_vs_predicho(pred_df["y_real"], pred_df["y_predicho"]), use_container_width=True)
    st.caption("Puntos más cercanos a la línea diagonal indican mejores predicciones.")

    st.plotly_chart(grafico_residuales(pred_df["error"], "Errores de predicción"), use_container_width=True)
    st.caption("Esta gráfica permite observar si los errores tienen patrones. Idealmente deberían fluctuar alrededor de cero sin estructura clara.")

    st.plotly_chart(grafico_histograma_interactivo(pred_df["error"], "Distribución de errores"), use_container_width=True)
    st.caption("La distribución de errores muestra si los errores están centrados alrededor de cero y si existen valores extremos.")

    imp_df = pd.DataFrame(res["importancia_variables"])
    if not imp_df.empty:
        st.plotly_chart(grafico_importancia_variables(imp_df), use_container_width=True)
        st.caption("La importancia de variables indica qué variables fueron más utilizadas por el modelo para construir sus predicciones.")

    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("Guardar resultados para resumen ejecutivo"):
            st.session_state["resultados_otras_regresiones"] = {**res, "fecha_guardado": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            st.success("Resultados guardados.")
    with c2:
        excel = preparar_excel_generico(res, {"Metricas": pd.DataFrame([res["metricas"]]), "Predicciones": pred_df, "Importancia": imp_df})
        st.download_button("Descargar Excel", excel, "resultados_otras_regresiones.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c3:
        st.download_button("Descargar JSON", preparar_json_descarga(res), "resultados_otras_regresiones.json", "application/json")
    with c4:
        figs = [
            ("Real vs. predicho", grafico_real_vs_predicho(pred_df["y_real"], pred_df["y_predicho"]), "Puntos cercanos a la diagonal indican mejores predicciones."),
            ("Errores de predicción", grafico_residuales(pred_df["error"], "Errores de predicción"), "Permite observar si los errores tienen patrones."),
            ("Distribución de errores", grafico_histograma_interactivo(pred_df["error"], "Distribución de errores"), "Ayuda a revisar si los errores están centrados alrededor de cero."),
        ]
        if not imp_df.empty:
            figs.append(("Importancia de variables", grafico_importancia_variables(imp_df), "Muestra qué variables pesan más en el modelo."))
        pdf_bytes = crear_pdf_modulo(
            "Informe del módulo: Otras regresiones",
            res,
            tablas=[("Métricas", pd.DataFrame([res["metricas"]])), ("Predicciones", pred_df), ("Importancia de variables", imp_df)],
            figuras=figs,
            notas=["El porcentaje de prueba corresponde a la proporción reservada para evaluación fuera de muestra.", "La estandarización vuelve comparables las escalas de las variables X."]
        )
        st.download_button("Descargar PDF", pdf_bytes, "informe_otras_regresiones.pdf", "application/pdf")
    with c5:
        if st.button("Eliminar resultados guardados"):
            st.session_state.pop("resultados_otras_regresiones", None)
            st.success("Resultados eliminados.")
