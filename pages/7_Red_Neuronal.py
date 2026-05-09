from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.carga_datos import cargar_datos_modulo, detectar_columnas_numericas
from utils.diagnosticos import matriz_confusion_df, metricas_clasificacion, metricas_regresion
from utils.exportacion import preparar_excel_generico, preparar_json_descarga
from utils.graficos import (
    grafico_arquitectura_red,
    grafico_histograma_interactivo,
    grafico_lineas_modelo,
    grafico_loss_curve,
    grafico_matriz_confusion,
    grafico_real_vs_predicho,
    grafico_residuales,
)
from utils.modelos import obtener_red_neuronal
from utils.pdf_reportes import crear_pdf_modulo
from utils.transformaciones import codificar_y
from utils.variables import capturar_descripcion_variables, descripcion_variables_df
from utils.estilos import mostrar_encabezado, caja_pedagogica


def tabla_comparativa_descriptiva_red(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compara estadísticos descriptivos de la variable objetivo real y la estimada por la red.
    """
    if pred_df.empty or "y_real" not in pred_df.columns or "y_predicho" not in pred_df.columns:
        return pd.DataFrame()

    comparacion = pred_df[["y_real", "y_predicho"]].rename(
        columns={
            "y_real": "Variable explicada histórica",
            "y_predicho": "Variable estimada por el modelo",
        }
    )

    tabla = comparacion.agg([
        "count",
        "mean",
        "median",
        "var",
        "std",
        "min",
        "max",
        "skew",
    ]).T.reset_index()

    tabla = tabla.rename(columns={
        "index": "serie",
        "count": "n_observaciones",
        "mean": "media",
        "median": "mediana",
        "var": "varianza",
        "std": "desviacion_estandar",
        "min": "minimo",
        "max": "maximo",
        "skew": "asimetria",
    })

    percentiles = comparacion.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T.reset_index()
    percentiles = percentiles.rename(columns={
        "index": "serie",
        0.01: "p1",
        0.05: "p5",
        0.25: "p25",
        0.50: "p50",
        0.75: "p75",
        0.95: "p95",
        0.99: "p99",
    })

    salida = tabla.merge(percentiles, on="serie", how="left")
    return salida


def tabla_descriptiva_residuales(residuales: pd.Series) -> pd.DataFrame:
    """
    Estadísticos descriptivos de los residuales de la red neuronal.
    """
    x = pd.to_numeric(residuales, errors="coerce").dropna()

    if x.empty:
        return pd.DataFrame()

    percentiles = x.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])

    return pd.DataFrame([{
        "n_observaciones": int(x.shape[0]),
        "media": float(x.mean()),
        "mediana": float(x.median()),
        "varianza": float(x.var(ddof=1)) if len(x) > 1 else np.nan,
        "desviacion_estandar": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
        "minimo": float(x.min()),
        "maximo": float(x.max()),
        "asimetria": float(x.skew()) if len(x) > 2 else np.nan,
        "p1": float(percentiles.loc[0.01]),
        "p5": float(percentiles.loc[0.05]),
        "p25": float(percentiles.loc[0.25]),
        "p50": float(percentiles.loc[0.50]),
        "p75": float(percentiles.loc[0.75]),
        "p95": float(percentiles.loc[0.95]),
        "p99": float(percentiles.loc[0.99]),
    }])


st.set_page_config(page_title="Red neuronal", page_icon="🧠", layout="wide")
mostrar_encabezado("🧠 Red neuronal", "Red neuronal multicapa para regresión o clasificación.")

caja_pedagogica("Permite configurar capas, neuronas, activación, iteraciones y tasa de aprendizaje.")

with st.expander("Guía rápida: ¿para qué sirven los parámetros de la red neuronal?", expanded=True):
    st.markdown(
        """
        **Tipo de problema:** define si la red se usará para predecir un valor continuo o una categoría.

        **Número de capas ocultas:** indica cuántas etapas internas de procesamiento tendrá la red.

        **Neuronas por capa:** define la capacidad de aprendizaje de cada capa.

        **Función de activación:** transforma la información dentro de la red. `relu` suele ser una opción práctica.

        **Iteraciones máximas:** número máximo de intentos de ajuste de los pesos de la red.

        **Tasa de aprendizaje:** controla el tamaño de los pasos durante el entrenamiento.

        **Número de datos finales para pronóstico:** en problemas de regresión, reserva las últimas observaciones para pronosticar con la red, sin usarlas en el entrenamiento.

        **Porcentaje de prueba:** en problemas de clasificación, reserva una parte de los datos para evaluar el modelo fuera de entrenamiento.
        """
    )

df, nombre_archivo, fuente_datos = cargar_datos_modulo("data/datos_red_neuronal.xlsx", "red_neuronal")
st.subheader("Vista previa de los datos seleccionados")
st.dataframe(df.head(20), use_container_width=True)

num_cols = detectar_columnas_numericas(df)
tipo = st.sidebar.selectbox("Tipo de problema", ["Regresión", "Clasificación"])
target_default = "Y_Regresion" if tipo == "Regresión" and "Y_Regresion" in df.columns else ("Clase" if "Clase" in df.columns else num_cols[0])
target = st.sidebar.selectbox("Variable objetivo", df.columns.tolist(), index=df.columns.tolist().index(target_default) if target_default in df.columns else 0)
x_cols = st.sidebar.multiselect("Variables explicativas numéricas", [c for c in num_cols if c != target], default=[c for c in num_cols if c != target][:5])

n_capas = st.sidebar.slider("Número de capas ocultas", 1, 3, 1)
neuronas = st.sidebar.slider("Neuronas por capa", 2, 50, 10)
activation = st.sidebar.selectbox("Función de activación", ["relu", "tanh", "logistic"])
max_iter = st.sidebar.slider("Iteraciones máximas", 100, 1000, 400, 50)
learning_rate = st.sidebar.number_input("Tasa de aprendizaje", min_value=0.0001, max_value=0.1, value=0.001, step=0.0005, format="%.4f")

if tipo == "Regresión":
    max_pronostico = max(0, len(df) - 10)
    n_pronostico = st.sidebar.number_input(
        "Número de datos finales para pronóstico",
        min_value=0,
        max_value=max_pronostico,
        value=min(20, max_pronostico),
        step=1,
        help="Estos datos finales no se usan para entrenar la red; se reservan para pronosticar la variable objetivo.",
    )
    test_size = None
else:
    n_pronostico = 0
    test_size = st.sidebar.slider("Porcentaje de prueba", 10, 40, 25) / 100

if not x_cols:
    st.warning("Seleccione variables explicativas.")
    st.stop()

descripcion_variables = capturar_descripcion_variables([target] + x_cols, "red_neuronal")

st.subheader("Esquema visual de la arquitectura seleccionada")
st.plotly_chart(grafico_arquitectura_red(len(x_cols), n_capas, neuronas, 1), use_container_width=True)
st.caption("El esquema muestra las variables de entrada, las capas ocultas configuradas y la salida del modelo. Si hay muchas neuronas, se muestra una representación simplificada.")

if "resultados_red_neuronal_actuales" not in st.session_state:
    st.session_state["resultados_red_neuronal_actuales"] = None

if st.button("Entrenar red neuronal"):
    datos = df[[target] + x_cols].dropna().reset_index(drop=True)

    if datos.empty:
        st.error("No hay datos válidos después de eliminar valores faltantes.")
        st.stop()

    hidden = tuple([neuronas] * n_capas)

    if tipo == "Regresión":
        if int(n_pronostico) >= len(datos) - 10:
            st.error("El número de datos para pronóstico es demasiado alto. Deben quedar suficientes datos para entrenar la red.")
            st.stop()

        if int(n_pronostico) > 0:
            datos_train = datos.iloc[:-int(n_pronostico)].copy()
            datos_fore = datos.iloc[-int(n_pronostico):].copy()
        else:
            datos_train = datos.copy()
            datos_fore = pd.DataFrame(columns=datos.columns)

        X_train = datos_train[x_cols]
        y_train = datos_train[target]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        modelo = obtener_red_neuronal(tipo, hidden, activation, max_iter, learning_rate)
        modelo.fit(X_train_s, y_train)

        y_pred_train = modelo.predict(X_train_s)
        metricas = metricas_regresion(y_train, y_pred_train)

        pred_df = pd.DataFrame({
            "observacion": range(len(y_pred_train)),
            "y_real": y_train.values,
            "y_predicho": y_pred_train,
        })
        pred_df["residual"] = pred_df["y_real"] - pred_df["y_predicho"]
        pred_df["periodo"] = "calibracion"

        tabla_descriptiva_comparativa = tabla_comparativa_descriptiva_red(pred_df)
        tabla_residuales = tabla_descriptiva_residuales(pred_df["residual"])

        pronostico_df = pd.DataFrame()
        if not datos_fore.empty:
            X_fore = datos_fore[x_cols]
            X_fore_s = scaler.transform(X_fore)
            y_fore_hat = modelo.predict(X_fore_s)

            pronostico_df = pd.DataFrame({
                "observacion": range(len(datos_train), len(datos_train) + len(datos_fore)),
                "y_real": datos_fore[target].values,
                "y_pronosticada": y_fore_hat,
                "error_pronostico": datos_fore[target].values - y_fore_hat,
                "periodo": "pronostico",
            })

        serie_modelo_df = pd.DataFrame({
            "observacion": pred_df["observacion"],
            "y_observada": pred_df["y_real"],
            "y_estimada": pred_df["y_predicho"],
            "y_pronosticada": np.nan,
        })

        if not pronostico_df.empty:
            tmp = pd.DataFrame({
                "observacion": pronostico_df["observacion"],
                "y_observada": pronostico_df["y_real"],
                "y_estimada": np.nan,
                "y_pronosticada": pronostico_df["y_pronosticada"],
            })
            serie_modelo_df = pd.concat([serie_modelo_df, tmp], ignore_index=True)

        cm_df = pd.DataFrame()
        loss_curve = getattr(modelo, "loss_curve_", [])

        st.session_state["resultados_red_neuronal_actuales"] = {
            "tipo_analisis": "Red neuronal",
            "fecha_ejecucion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "archivo_usado": nombre_archivo,
            "fuente_datos": fuente_datos,
            "tipo_problema": tipo,
            "variable_objetivo": target,
            "variables_explicativas": x_cols,
            "descripcion_variables": descripcion_variables,
            "arquitectura": str(hidden),
            "activation": activation,
            "max_iter": max_iter,
            "learning_rate_init": learning_rate,
            "n_pronostico": int(n_pronostico),
            "filas_calibracion": len(datos_train),
            "filas_pronostico": len(datos_fore),
            "metricas": metricas,
            "tabla_descriptiva_comparativa": tabla_descriptiva_comparativa.to_dict(orient="records"),
            "tabla_residuales": tabla_residuales.to_dict(orient="records"),
            "predicciones": pred_df.to_dict(orient="records"),
            "pronostico": pronostico_df.to_dict(orient="records"),
            "serie_modelo": serie_modelo_df.to_dict(orient="records"),
            "matriz_confusion": [],
            "loss_curve": list(map(float, loss_curve)) if loss_curve is not None else [],
        }

    else:
        X = datos[x_cols]
        y, le = codificar_y(datos[target])

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=test_size, random_state=42)

        modelo = obtener_red_neuronal(tipo, hidden, activation, max_iter, learning_rate)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        metricas = metricas_clasificacion(y_test, y_pred)
        pred_df = pd.DataFrame({"observacion": range(len(y_pred)), "y_real": y_test, "y_predicho": y_pred})
        cm_df = matriz_confusion_df(y_test, y_pred)
        loss_curve = getattr(modelo, "loss_curve_", [])

        st.session_state["resultados_red_neuronal_actuales"] = {
            "tipo_analisis": "Red neuronal",
            "fecha_ejecucion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "archivo_usado": nombre_archivo,
            "fuente_datos": fuente_datos,
            "tipo_problema": tipo,
            "variable_objetivo": target,
            "variables_explicativas": x_cols,
            "descripcion_variables": descripcion_variables,
            "arquitectura": str(hidden),
            "activation": activation,
            "max_iter": max_iter,
            "learning_rate_init": learning_rate,
            "test_size": test_size,
            "metricas": metricas,
            "tabla_descriptiva_comparativa": [],
            "tabla_residuales": [],
            "predicciones": pred_df.to_dict(orient="records"),
            "pronostico": [],
            "serie_modelo": [],
            "matriz_confusion": cm_df.reset_index().to_dict(orient="records"),
            "loss_curve": list(map(float, loss_curve)) if loss_curve is not None else [],
        }

res = st.session_state["resultados_red_neuronal_actuales"]
if res:
    st.subheader("Métricas")
    st.json(res["metricas"])
    st.caption("Las métricas resumen el desempeño del modelo. En regresión se revisan R², MAE y RMSE; en clasificación, accuracy, precision, recall y F1.")

    pred_df = pd.DataFrame(res["predicciones"])
    pronostico_df = pd.DataFrame(res.get("pronostico", []))
    serie_modelo_df = pd.DataFrame(res.get("serie_modelo", []))
    tabla_descriptiva_comparativa = pd.DataFrame(res.get("tabla_descriptiva_comparativa", []))
    tabla_residuales = pd.DataFrame(res.get("tabla_residuales", []))
    desc_df = descripcion_variables_df(res.get("descripcion_variables", {}))

    if res["tipo_problema"] == "Regresión":
        st.subheader("Comparación descriptiva: variable histórica vs. estimada")
        st.dataframe(tabla_descriptiva_comparativa, use_container_width=True)
        st.caption(
            "Esta tabla compara los principales estadísticos descriptivos de la variable explicada observada "
            "y la variable estimada por la red neuronal. Sirve para revisar si el modelo reproduce niveles, "
            "dispersión y percentiles de la serie histórica."
        )

        st.subheader("Variable objetivo histórica vs. resultado de la red neuronal")
        st.plotly_chart(
            grafico_lineas_modelo(
                pred_df,
                "observacion",
                ["y_real", "y_predicho"],
                "Variable objetivo histórica vs. resultado de la red neuronal en calibración",
            ),
            use_container_width=True,
        )
        st.caption("Esta gráfica compara la variable objetivo real con la salida de la red neuronal en el periodo de calibración.")

        st.plotly_chart(grafico_real_vs_predicho(pred_df["y_real"], pred_df["y_predicho"]), use_container_width=True)
        st.caption("Mientras más cerca estén los puntos de la línea diagonal, mejor será la capacidad predictiva del modelo.")

        st.subheader("Residuales de la red neuronal")
        st.dataframe(pred_df[["observacion", "y_real", "y_predicho", "residual"]], use_container_width=True)
        st.write("**Estadísticos descriptivos de residuales**")
        st.dataframe(tabla_residuales, use_container_width=True)

        st.plotly_chart(grafico_residuales(pred_df["residual"], "Residuales de la red neuronal"), use_container_width=True)
        st.caption("Los residuales muestran la diferencia entre el valor observado y el valor estimado por la red neuronal.")

        st.plotly_chart(grafico_histograma_interactivo(pred_df["residual"], "Distribución de residuales de la red neuronal"), use_container_width=True)
        st.caption("El histograma permite revisar si los errores están concentrados alrededor de cero y si existen valores extremos.")

        st.subheader("Pronóstico con los datos finales reservados")
        if pronostico_df.empty:
            st.info("No se reservaron datos finales para pronóstico.")
        else:
            st.dataframe(pronostico_df, use_container_width=True)
            st.plotly_chart(
                grafico_lineas_modelo(
                    serie_modelo_df,
                    "observacion",
                    ["y_observada", "y_estimada", "y_pronosticada"],
                    "Calibración y pronóstico con red neuronal",
                ),
                use_container_width=True,
            )
            st.caption("La línea pronosticada corresponde a los datos finales reservados que no se usaron para entrenar la red neuronal.")

    else:
        cm_records = res["matriz_confusion"]
        if cm_records:
            cm_df = pd.DataFrame(cm_records).set_index("index")
            st.plotly_chart(grafico_matriz_confusion(cm_df), use_container_width=True)
            st.caption("La matriz de confusión muestra aciertos y errores de clasificación.")

    if not desc_df.empty:
        st.subheader("Descripción de variables")
        st.dataframe(desc_df, use_container_width=True)

    if res["loss_curve"]:
        st.plotly_chart(grafico_loss_curve(res["loss_curve"]), use_container_width=True)
        st.caption("La curva de pérdida muestra cómo cambia el error durante el entrenamiento.")

    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("Guardar resultados para resumen ejecutivo"):
            st.session_state["resultados_red_neuronal"] = {**res, "fecha_guardado": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            st.success("Resultados guardados.")

    with c2:
        excel = preparar_excel_generico(
            res,
            {
                "Descripcion_variables": desc_df,
                "Comparacion_descriptiva": tabla_descriptiva_comparativa,
                "Estadisticos_residuales": tabla_residuales,
                "Metricas": pd.DataFrame([res["metricas"]]),
                "Predicciones": pred_df,
                "Pronostico": pronostico_df,
                "Serie_modelo": serie_modelo_df,
                "Loss": pd.DataFrame({"loss": res["loss_curve"]}),
            },
        )
        st.download_button("Descargar Excel", excel, "resultados_red_neuronal.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with c3:
        st.download_button("Descargar JSON", preparar_json_descarga(res), "resultados_red_neuronal.json", "application/json")

    with c4:
        figs = []
        if res["tipo_problema"] == "Regresión":
            figs.append((
                "Variable objetivo real vs. resultado de la red",
                grafico_lineas_modelo(pred_df, "observacion", ["y_real", "y_predicho"], "Variable objetivo real vs. resultado de la red"),
                "Muestra la comparación en el periodo de calibración.",
            ))
            figs.append((
                "Real vs. predicho",
                grafico_real_vs_predicho(pred_df["y_real"], pred_df["y_predicho"]),
                "Puntos cercanos a la diagonal indican mejor predicción.",
            ))
            figs.append((
                "Residuales de la red neuronal",
                grafico_residuales(pred_df["residual"], "Residuales de la red neuronal"),
                "Muestra el error de estimación por observación.",
            ))
            figs.append((
                "Distribución de residuales",
                grafico_histograma_interactivo(pred_df["residual"], "Distribución de residuales de la red neuronal"),
                "Permite revisar concentración y valores extremos de los errores.",
            ))
            if not serie_modelo_df.empty:
                figs.append((
                    "Calibración y pronóstico",
                    grafico_lineas_modelo(serie_modelo_df, "observacion", ["y_observada", "y_estimada", "y_pronosticada"], "Calibración y pronóstico con red neuronal"),
                    "Muestra la salida del modelo en calibración y en el periodo reservado para pronóstico.",
                ))
        else:
            cm_records = res["matriz_confusion"]
            if cm_records:
                cm_df_pdf = pd.DataFrame(cm_records).set_index("index")
                figs.append((
                    "Matriz de confusión",
                    grafico_matriz_confusion(cm_df_pdf),
                    "La diagonal principal muestra aciertos.",
                ))

        if res["loss_curve"]:
            figs.append((
                "Evolución de la pérdida",
                grafico_loss_curve(res["loss_curve"]),
                "Una pérdida decreciente indica aprendizaje durante el entrenamiento.",
            ))

        pdf_bytes = crear_pdf_modulo(
            "Informe del módulo: Red neuronal",
            res,
            tablas=[
                ("Descripción de variables", desc_df),
                ("Comparación descriptiva", tabla_descriptiva_comparativa),
                ("Estadísticos de residuales", tabla_residuales),
                ("Métricas", pd.DataFrame([res["metricas"]]),),
                ("Predicciones", pred_df),
                ("Pronóstico", pronostico_df),
                ("Serie modelo", serie_modelo_df),
                ("Loss curve", pd.DataFrame({"loss": res["loss_curve"]})),
            ],
            figuras=figs,
            notas=[
                "La comparación descriptiva permite revisar si la red reproduce niveles, dispersión y percentiles de la variable histórica.",
                "Los residuales miden la diferencia entre el valor observado y el valor estimado por la red neuronal.",
                "El pronóstico usa los datos finales reservados que no se emplean para entrenar la red.",
                "Los resultados son pedagógicos y requieren validación para cualquier uso real.",
            ],
        )
        st.download_button("Descargar PDF", pdf_bytes, "informe_red_neuronal.pdf", "application/pdf")

    with c5:
        if st.button("Eliminar resultados guardados"):
            st.session_state.pop("resultados_red_neuronal", None)
            st.success("Resultados eliminados.")
