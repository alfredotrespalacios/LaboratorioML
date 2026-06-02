from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from utils.carga_datos import cargar_datos_modulo, detectar_columnas_fecha, detectar_columnas_numericas
from utils.exportacion import preparar_excel_generico, preparar_json_descarga
from utils.graficos import grafico_histograma_interactivo, grafico_lineas_modelo, grafico_real_vs_predicho, grafico_residuales
from utils.pdf_reportes import crear_pdf_modulo
from utils.pronostico import (
    MODO_EVALUACION,
    MODO_VALORES_NO_OBSERVADOS,
    agregar_errores_pronostico,
    descripcion_modo_pronostico,
)
from utils.regresion_helpers import (
    metricas_pronostico_ext,
    metricas_regresion_ext,
    preparar_tabla_doble_escala,
    tabla_interpretacion_error_medio,
    tabla_modo_pronostico,
)
from utils.transformaciones import aplicar_log_si_corresponde, ordenar_por_fecha
from utils.variables import capturar_descripcion_variables, descripcion_variables_df
from utils.estilos import mostrar_encabezado, caja_pedagogica

st.set_page_config(page_title="KNN Regresión", page_icon="📍", layout="wide")
mostrar_encabezado("📍 KNN Regresión", "Modelo de vecinos más cercanos para predecir una variable continua.")

caja_pedagogica(
    "KNN Regresión predice una variable continua buscando las K observaciones más cercanas según las variables X. "
    "La predicción se calcula como el promedio simple de los valores Y de esos vecinos."
)

with st.expander("Explicación pedagógica del método KNN Regresión", expanded=True):
    st.markdown(
        """
        **KNN Regresión** no estima coeficientes como una regresión lineal. Para pronosticar una observación, busca las **K observaciones más parecidas** en las variables explicativas y promedia sus valores de Y.

        **K:** número de vecinos usados en la predicción. Un K muy bajo puede ajustarse demasiado al ruido; un K muy alto puede suavizar en exceso la relación.

        **Distancia euclidiana:** mide qué tan cerca están dos observaciones en el espacio de las variables X.

        **Estandarización:** en KNN es especialmente importante, porque las variables con mayor escala pueden dominar la distancia. Por eso se recomienda estandarizar X, aunque el estudiante puede comparar ambos casos.

        **Transformaciones logarítmicas:** si se usa ln(Y), el modelo aprende y predice en logaritmos. En ese caso, el módulo también muestra resultados retransformados a nivel mediante exp(predicción).
        """
    )

df, nombre_archivo, fuente_datos = cargar_datos_modulo("data/datos_knn_regresion.xlsx", "knn_regresion")
filas_originales = len(df)

st.subheader("Vista previa de los datos seleccionados")
st.dataframe(df.head(20), use_container_width=True)

num_cols = detectar_columnas_numericas(df)
if len(num_cols) < 2:
    st.error("Se requieren al menos dos columnas numéricas.")
    st.stop()

st.sidebar.header("2. Variables")
y_col = st.sidebar.selectbox("Variable objetivo Y", num_cols, index=0)
x_cols = st.sidebar.multiselect("Variables explicativas X", [c for c in num_cols if c != y_col], default=[c for c in num_cols if c != y_col][:3])
if not x_cols:
    st.warning("Seleccione al menos una variable explicativa.")
    st.stop()

st.sidebar.header("3. Orden y modo de pronóstico")
columnas_fecha = detectar_columnas_fecha(df)
columna_fecha = st.sidebar.selectbox("Columna de fecha u orden", ["Ninguna"] + columnas_fecha)
modo_pronostico = st.sidebar.radio("Modo de pronóstico", [MODO_EVALUACION, MODO_VALORES_NO_OBSERVADOS])
info_modo = descripcion_modo_pronostico(modo_pronostico)

if modo_pronostico == MODO_EVALUACION:
    max_pronostico = max(0, len(df) - 10)
    n_pronostico = st.sidebar.number_input("Número de datos finales para evaluación de pronóstico", min_value=0, max_value=max_pronostico, value=min(20, max_pronostico), step=1)
else:
    n_pronostico = 0
    st.sidebar.info("Este modo usa filas donde Y está vacía y las X están disponibles.")

st.info(f"**Modo seleccionado:** {info_modo['modo_pronostico']}\n\n**Uso:** {info_modo['uso_modo_pronostico']}\n\n**Implicación:** {info_modo['implicacion_modo_pronostico']}")

st.sidebar.header("4. Transformaciones y configuración KNN")
usar_log_y = st.sidebar.radio(f"Transformación de {y_col}", ["Nivel", "ln(Y)"], horizontal=True) == "ln(Y)"
transformaciones_x = {}
for x in x_cols:
    transformaciones_x[x] = st.sidebar.radio(f"Transformación de {x}", ["Nivel", f"ln({x})"], horizontal=True, key=f"knn_log_{x}") != "Nivel"

estandarizar_x = st.sidebar.checkbox("Estandarizar variables X", value=True)
k_vecinos = st.sidebar.slider("Número de vecinos K", 1, 50, 5)
k_max_grafico = st.sidebar.slider("K máximo para gráfico de error", 5, 60, 30)

descripcion_variables = capturar_descripcion_variables([y_col] + x_cols, "knn_regresion")

if "resultados_knn_regresion_actuales" not in st.session_state:
    st.session_state["resultados_knn_regresion_actuales"] = None

def preparar_datos(df_base: pd.DataFrame):
    df_ordenado = ordenar_por_fecha(df_base, columna_fecha)
    datos = pd.DataFrame(index=df_ordenado.index)
    y_trans, y_nombre = aplicar_log_si_corresponde(df_ordenado, y_col, usar_log_y)
    datos[y_nombre] = y_trans
    x_nombres = []
    for x in x_cols:
        x_trans, x_nombre = aplicar_log_si_corresponde(df_ordenado, x, transformaciones_x[x])
        datos[x_nombre] = x_trans
        x_nombres.append(x_nombre)
    datos["_eje_x"] = pd.to_datetime(df_ordenado[columna_fecha], errors="coerce").astype(str) if columna_fecha != "Ninguna" else list(range(len(datos)))
    datos["_y_original"] = df_ordenado[y_col].values
    return datos.replace([np.inf, -np.inf], np.nan), y_nombre, x_nombres

def fit_predict(X_train, y_train, X_pred, k, estandarizar):
    k = int(min(max(1, k), len(X_train)))
    scaler = None
    if estandarizar:
        scaler = StandardScaler()
        X_train_model = scaler.fit_transform(X_train)
        X_pred_model = scaler.transform(X_pred)
    else:
        X_train_model = X_train.values
        X_pred_model = X_pred.values
    modelo = KNeighborsRegressor(n_neighbors=k, metric="euclidean", weights="uniform")
    modelo.fit(X_train_model, y_train)
    return modelo, scaler, X_train_model, X_pred_model, modelo.predict(X_pred_model)

if st.button("Entrenar KNN Regresión"):
    datos, y_nombre, x_nombres = preparar_datos(df)
    if modo_pronostico == MODO_EVALUACION:
        datos_validos = datos.dropna(subset=[y_nombre] + x_nombres).copy()
        if int(n_pronostico) >= len(datos_validos) - 5:
            st.error("El número de datos para pronóstico es demasiado alto.")
            st.stop()
        train_df = datos_validos.iloc[:-int(n_pronostico)].copy() if int(n_pronostico) > 0 else datos_validos.copy()
        fore_df = datos_validos.iloc[-int(n_pronostico):].copy() if int(n_pronostico) > 0 else pd.DataFrame(columns=datos.columns)
    else:
        train_df = datos.dropna(subset=[y_nombre] + x_nombres).copy()
        fore_df = datos[datos[y_nombre].isna()].dropna(subset=x_nombres).copy()
        if fore_df.empty:
            st.warning("No se encontraron filas con Y vacía y variables X completas para pronosticar valores no observados.")

    if len(train_df) < 3:
        st.error("No hay suficientes datos para entrenar KNN.")
        st.stop()

    k_usado = int(min(max(1, k_vecinos), len(train_df)))
    X_train = train_df[x_nombres]
    y_train = train_df[y_nombre]

    modelo, scaler, X_train_model, _, y_pred_train = fit_predict(X_train, y_train, X_train, k_usado, estandarizar_x)

    pred_df = pd.DataFrame({
        "observacion": train_df.index,
        "eje_x": train_df["_eje_x"].values,
        "y_real": y_train.values,
        "y_predicho": y_pred_train,
        "residual": y_train.values - y_pred_train,
        "periodo": "calibracion",
    })
    pred_df = preparar_tabla_doble_escala(pred_df, usar_log_y, "prediccion")
    metricas = metricas_regresion_ext(pred_df["y_real"], pred_df["y_predicho"])
    metricas_nivel = metricas_regresion_ext(pred_df["y_real_nivel"], pred_df["y_predicho_nivel"]) if usar_log_y else {}

    pronostico_df = pd.DataFrame()
    if not fore_df.empty:
        _, _, _, _, y_fore = fit_predict(X_train, y_train, fore_df[x_nombres], k_usado, estandarizar_x)
        pronostico_df = pd.DataFrame({
            "observacion": fore_df.index,
            "eje_x": fore_df["_eje_x"].values,
            "y_real": fore_df[y_nombre].values,
            "y_pronosticada": y_fore,
            "periodo": "evaluacion_pronostico" if modo_pronostico == MODO_EVALUACION else "valor_no_observado",
        })
        if modo_pronostico == MODO_EVALUACION:
            pronostico_df = agregar_errores_pronostico(pronostico_df, "y_real", "y_pronosticada")
        pronostico_df = preparar_tabla_doble_escala(pronostico_df, usar_log_y, "pronostico")

    metricas_fore = metricas_pronostico_ext(pronostico_df.get("y_real", []), pronostico_df.get("y_pronosticada", [])) if modo_pronostico == MODO_EVALUACION and not pronostico_df.empty else {}
    metricas_fore_nivel = metricas_pronostico_ext(pronostico_df["y_real_nivel"], pronostico_df["y_pronosticada_nivel"]) if usar_log_y and modo_pronostico == MODO_EVALUACION and not pronostico_df.empty and "y_real_nivel" in pronostico_df.columns else {}

    serie_modelo_df = pd.DataFrame({"eje_x": pred_df["eje_x"], "y_observada": pred_df["y_real"], "y_estimada": pred_df["y_predicho"], "y_pronosticada": np.nan})
    if not pronostico_df.empty:
        tmp = pd.DataFrame({"eje_x": pronostico_df["eje_x"], "y_observada": pronostico_df["y_real"], "y_estimada": np.nan, "y_pronosticada": pronostico_df["y_pronosticada"]})
        serie_modelo_df = pd.concat([serie_modelo_df, tmp], ignore_index=True)

    serie_modelo_nivel_df = pd.DataFrame()
    if usar_log_y:
        serie_modelo_nivel_df = pd.DataFrame({"eje_x": pred_df["eje_x"], "y_observada_nivel": pred_df["y_real_nivel"], "y_estimada_nivel": pred_df["y_predicho_nivel"], "y_pronosticada_nivel": np.nan})
        if not pronostico_df.empty and "y_pronosticada_nivel" in pronostico_df.columns:
            tmpn = pd.DataFrame({"eje_x": pronostico_df["eje_x"], "y_observada_nivel": pronostico_df.get("y_real_nivel", np.nan), "y_estimada_nivel": np.nan, "y_pronosticada_nivel": pronostico_df["y_pronosticada_nivel"]})
            serie_modelo_nivel_df = pd.concat([serie_modelo_nivel_df, tmpn], ignore_index=True)

    error_k_df = pd.DataFrame()
    if modo_pronostico == MODO_EVALUACION and not fore_df.empty:
        vals = []
        for k in range(1, int(min(k_max_grafico, len(train_df))) + 1):
            _, _, _, _, yk = fit_predict(X_train, y_train, fore_df[x_nombres], k, estandarizar_x)
            vals.append({"K": k, **metricas_pronostico_ext(fore_df[y_nombre].values, yk)})
        error_k_df = pd.DataFrame(vals)

    # Neighbor example uses default reference. User can select display later.
    ref_pool = fore_df if not fore_df.empty else train_df.tail(1)
    ref_options = []
    if not fore_df.empty:
        ref_options += [{"tipo": "Primera observación de pronóstico", "idx": int(fore_df.index[0])}]
    ref_options += [{"tipo": "Última observación de calibración", "idx": int(train_df.index[-1])}]

    st.session_state["resultados_knn_regresion_actuales"] = {
        "tipo_analisis": "KNN Regresión",
        "fecha_ejecucion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "archivo_usado": nombre_archivo,
        "fuente_datos": fuente_datos,
        "filas_originales": filas_originales,
        "filas_calibracion": len(train_df),
        "filas_pronostico": len(fore_df),
        "variable_objetivo_original": y_col,
        "variable_objetivo_usada": y_nombre,
        "variables_explicativas_originales": x_cols,
        "variables_explicativas_usadas": x_nombres,
        "descripcion_variables": descripcion_variables,
        "transformacion_y": "logaritmo_natural" if usar_log_y else "nivel",
        "transformaciones_x": {k: ("logaritmo_natural" if v else "nivel") for k, v in transformaciones_x.items()},
        "K": k_usado,
        "distancia": "euclidiana",
        "ponderacion": "promedio_simple",
        "estandarizacion_x": estandarizar_x,
        **info_modo,
        "n_pronostico": int(n_pronostico) if modo_pronostico == MODO_EVALUACION else None,
        "metricas": metricas,
        "metricas_nivel": metricas_nivel,
        "metricas_pronostico": metricas_fore,
        "metricas_pronostico_nivel": metricas_fore_nivel,
        "predicciones": pred_df.to_dict(orient="records"),
        "pronostico": pronostico_df.to_dict(orient="records"),
        "serie_modelo": serie_modelo_df.to_dict(orient="records"),
        "serie_modelo_nivel": serie_modelo_nivel_df.to_dict(orient="records"),
        "error_por_k": error_k_df.to_dict(orient="records"),
        "referencias_vecinos": ref_options,
        "_train_df": train_df.to_dict(orient="records"),
        "_fore_df": fore_df.to_dict(orient="records"),
        "_x_nombres": x_nombres,
        "_y_nombre": y_nombre,
    }

res = st.session_state["resultados_knn_regresion_actuales"]
if res:
    st.subheader("Modo de pronóstico usado")
    st.info(f"**{res.get('modo_pronostico')}**\n\n**Uso:** {res.get('uso_modo_pronostico')}\n\n**Implicación:** {res.get('implicacion_modo_pronostico')}")

    pred_df = pd.DataFrame(res["predicciones"])
    pronostico_df = pd.DataFrame(res.get("pronostico", []))
    serie_modelo_df = pd.DataFrame(res.get("serie_modelo", []))
    serie_modelo_nivel_df = pd.DataFrame(res.get("serie_modelo_nivel", []))
    error_k_df = pd.DataFrame(res.get("error_por_k", []))
    desc_df = descripcion_variables_df(res.get("descripcion_variables", {}))
    metricas_df = pd.DataFrame([res["metricas"]])
    metricas_nivel_df = pd.DataFrame([res.get("metricas_nivel", {})])
    metricas_pronostico_df = pd.DataFrame([res.get("metricas_pronostico", {})])
    metricas_pronostico_nivel_df = pd.DataFrame([res.get("metricas_pronostico_nivel", {})])

    st.subheader("Configuración del modelo")
    st.dataframe(pd.DataFrame([{"K": res.get("K"), "distancia": res.get("distancia"), "ponderacion": res.get("ponderacion"), "estandarizacion_x": res.get("estandarizacion_x"), "transformacion_y": res.get("transformacion_y")}]), use_container_width=True)

    st.subheader("Métricas de calibración")
    st.dataframe(metricas_df, use_container_width=True)
    if res.get("metricas_nivel"):
        st.write("**Métricas de calibración en nivel original de Y**")
        st.dataframe(metricas_nivel_df, use_container_width=True)

    st.subheader("Variable explicada real vs. predicha en calibración")
    st.plotly_chart(grafico_lineas_modelo(pred_df, "eje_x", ["y_real", "y_predicho"], "KNN: variable explicada real vs. predicha"), use_container_width=True)
    st.plotly_chart(grafico_real_vs_predicho(pred_df["y_real"], pred_df["y_predicho"]), use_container_width=True)

    st.subheader("Residuales")
    resid_cols = [c for c in ["observacion", "eje_x", "y_real", "y_predicho", "residual", "y_real_nivel", "y_predicho_nivel", "residual_nivel"] if c in pred_df.columns]
    st.dataframe(pred_df[resid_cols], use_container_width=True)
    st.plotly_chart(grafico_residuales(pred_df["residual"], "Residuales KNN"), use_container_width=True)
    st.plotly_chart(grafico_histograma_interactivo(pred_df["residual"], "Distribución de residuales KNN"), use_container_width=True)

    st.subheader("Pronóstico")
    if pronostico_df.empty:
        st.info("No hay filas disponibles para pronóstico según el modo seleccionado.")
    else:
        st.dataframe(pronostico_df, use_container_width=True)
        if res.get("metricas_pronostico"):
            st.write("**Métricas de evaluación de pronóstico en la escala usada por el modelo**")
            st.dataframe(metricas_pronostico_df, use_container_width=True)
            if "Error_medio_pronostico" in metricas_pronostico_df.columns:
                st.dataframe(tabla_interpretacion_error_medio(metricas_pronostico_df.loc[0, "Error_medio_pronostico"], "Error medio de pronóstico"), use_container_width=True)
        if res.get("metricas_pronostico_nivel"):
            st.write("**Métricas de evaluación de pronóstico en nivel original de Y**")
            st.dataframe(metricas_pronostico_nivel_df, use_container_width=True)
        st.plotly_chart(grafico_lineas_modelo(serie_modelo_df, "eje_x", ["y_observada", "y_estimada", "y_pronosticada"], "KNN: calibración y pronóstico"), use_container_width=True)
        if not serie_modelo_nivel_df.empty:
            st.plotly_chart(grafico_lineas_modelo(serie_modelo_nivel_df, "eje_x", ["y_observada_nivel", "y_estimada_nivel", "y_pronosticada_nivel"], "KNN: calibración y pronóstico en nivel original"), use_container_width=True)

    if not error_k_df.empty:
        st.subheader("Error según número de vecinos K")
        st.line_chart(error_k_df.set_index("K")[["RMSE_pronostico"]])
        st.dataframe(error_k_df, use_container_width=True)
        st.caption("La tabla compara el error de pronóstico para distintos valores de K. Incluye error medio, MAE, RMSE, MAPE y U de Theil cuando aplica.")

    with st.expander("Ejemplo pedagógico: vecinos más cercanos usados por KNN", expanded=True):
        st.write("Esta sección aclara **de qué observación** se están mostrando los vecinos y cómo se obtiene el pronóstico KNN como promedio simple de Y de esos vecinos.")

        train_ref_df = pd.DataFrame(res.get("_train_df", []))
        fore_ref_df = pd.DataFrame(res.get("_fore_df", []))
        x_ref_cols = res.get("_x_nombres", [])
        y_ref_col = res.get("_y_nombre")

        opciones = []
        if not fore_ref_df.empty:
            opciones.append("Primera observación de pronóstico")
        opciones.append("Última observación de calibración")
        opciones.append("Seleccionar manualmente por número de fila")

        opcion_ref = st.selectbox("Seleccione observación de referencia", opciones)

        if opcion_ref == "Primera observación de pronóstico" and not fore_ref_df.empty:
            ref_row = fore_ref_df.iloc[[0]]
            etiqueta_ref = f"Primera observación de pronóstico. Índice original: {ref_row.index[0]}"
        elif opcion_ref == "Última observación de calibración":
            ref_row = train_ref_df.iloc[[-1]]
            etiqueta_ref = f"Última observación de calibración. Índice original: {ref_row.index[0]}"
        else:
            indices_disponibles = list(train_ref_df.index)
            if not fore_ref_df.empty:
                indices_disponibles += list(fore_ref_df.index)
            indice_manual = st.selectbox("Número de fila disponible", indices_disponibles)
            if indice_manual in fore_ref_df.index:
                ref_row = fore_ref_df.loc[[indice_manual]]
                etiqueta_ref = f"Observación manual de pronóstico. Índice original: {indice_manual}"
            else:
                ref_row = train_ref_df.loc[[indice_manual]]
                etiqueta_ref = f"Observación manual de calibración. Índice original: {indice_manual}"

        st.write(f"**Observación de referencia:** {etiqueta_ref}")
        st.write("**Valores X de la observación de referencia**")
        st.dataframe(ref_row[x_ref_cols], use_container_width=True)

        try:
            X_train_ref = train_ref_df[x_ref_cols]
            y_train_ref = train_ref_df[y_ref_col]
            X_ref = ref_row[x_ref_cols]

            if res.get("estandarizacion_x"):
                scaler_ref = StandardScaler()
                X_train_model_ref = scaler_ref.fit_transform(X_train_ref)
                X_ref_model = scaler_ref.transform(X_ref)
            else:
                X_train_model_ref = X_train_ref.values
                X_ref_model = X_ref.values

            nn = NearestNeighbors(n_neighbors=int(res.get("K")), metric="euclidean")
            nn.fit(X_train_model_ref)
            distancias, indices_vecinos = nn.kneighbors(X_ref_model)

            vecinos_df = train_ref_df.iloc[indices_vecinos[0]][["_eje_x", y_ref_col] + x_ref_cols].copy().reset_index(drop=True)
            vecinos_df.insert(0, "vecino", range(1, len(vecinos_df) + 1))
            vecinos_df.insert(1, "distancia_euclidiana", distancias[0])

            resumen_vecinos = pd.DataFrame([{
                "K_usado": int(res.get("K")),
                "Y_minima_vecinos": float(y_train_ref.iloc[indices_vecinos[0]].min()),
                "Y_promedio_vecinos": float(y_train_ref.iloc[indices_vecinos[0]].mean()),
                "Y_maxima_vecinos": float(y_train_ref.iloc[indices_vecinos[0]].max()),
                "distancia_promedio": float(distancias[0].mean()),
                "pronostico_KNN": float(y_train_ref.iloc[indices_vecinos[0]].mean()),
            }])

            if res.get("transformacion_y") == "logaritmo_natural":
                resumen_vecinos["pronostico_KNN_nivel"] = np.exp(resumen_vecinos["pronostico_KNN"])

            st.write("**Vecinos históricos más cercanos**")
            st.dataframe(vecinos_df, use_container_width=True)
            st.write("**Resumen del pronóstico por vecinos**")
            st.dataframe(resumen_vecinos, use_container_width=True)
        except Exception as exc:
            st.warning(f"No fue posible calcular dinámicamente los vecinos para esta observación: {exc}")

    if not desc_df.empty:
        st.subheader("Descripción de variables")
        st.dataframe(desc_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Gestión de resultados")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("Guardar resultados para resumen ejecutivo"):
            # remove private helper fields
            clean = {k:v for k,v in res.items() if not k.startswith("_")}
            st.session_state["resultados_knn_regresion"] = {**clean, "fecha_guardado": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            st.success("Resultados guardados.")
    clean_res = {k:v for k,v in res.items() if not k.startswith("_")}
    with c2:
        excel = preparar_excel_generico(clean_res, {
            "Descripcion_variables": desc_df,
            "Configuracion": pd.DataFrame([{"K": res.get("K"), "distancia": res.get("distancia"), "ponderacion": res.get("ponderacion"), "estandarizacion_x": res.get("estandarizacion_x"), "transformacion_y": res.get("transformacion_y")}]),
            "Modo_pronostico": tabla_modo_pronostico(res),
            "Metricas": metricas_df,
            "Metricas_nivel": metricas_nivel_df,
            "Metricas_pronostico": metricas_pronostico_df,
            "Metricas_pronostico_nivel": metricas_pronostico_nivel_df,
            "Predicciones": pred_df,
            "Pronostico": pronostico_df,
            "Serie_modelo": serie_modelo_df,
            "Serie_modelo_nivel": serie_modelo_nivel_df,
            "Error_por_K": error_k_df,
        })
        st.download_button("Descargar Excel", excel, "resultados_knn_regresion.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c3:
        st.download_button("Descargar JSON", preparar_json_descarga(clean_res), "resultados_knn_regresion.json", "application/json")
    with c4:
        figs = [
            ("Variable explicada real vs. predicha", grafico_lineas_modelo(pred_df, "eje_x", ["y_real", "y_predicho"], "KNN: variable explicada real vs. predicha"), "Compara la variable real con la predicción KNN en calibración."),
            ("Residuales KNN", grafico_residuales(pred_df["residual"], "Residuales KNN"), "Muestra los errores de calibración del modelo KNN."),
            ("Calibración y pronóstico", grafico_lineas_modelo(serie_modelo_df, "eje_x", ["y_observada", "y_estimada", "y_pronosticada"], "KNN: calibración y pronóstico"), "Muestra calibración y pronóstico según el modo seleccionado."),
        ]
        if not serie_modelo_nivel_df.empty:
            figs.append(("Calibración y pronóstico en nivel original", grafico_lineas_modelo(serie_modelo_nivel_df, "eje_x", ["y_observada_nivel", "y_estimada_nivel", "y_pronosticada_nivel"], "KNN: nivel original"), "Resultados retransformados a nivel original cuando se usa ln(Y)."))
        pdf_bytes = crear_pdf_modulo(
            "Informe del módulo: KNN Regresión",
            clean_res,
            tablas=[
                ("Descripción de variables", desc_df),
                ("Configuración", pd.DataFrame([{"K": res.get("K"), "distancia": res.get("distancia"), "ponderacion": res.get("ponderacion"), "estandarizacion_x": res.get("estandarizacion_x"), "transformacion_y": res.get("transformacion_y")}])),
                ("Modo de pronóstico", tabla_modo_pronostico(res)),
                ("Métricas", metricas_df),
                ("Métricas en nivel", metricas_nivel_df),
                ("Métricas de pronóstico", metricas_pronostico_df),
                ("Métricas de pronóstico en nivel", metricas_pronostico_nivel_df),
                ("Predicciones", pred_df),
                ("Pronóstico", pronostico_df),
                ("Error por K", error_k_df),
            ],
            figuras=figs,
            notas=["KNN predice por similitud usando el promedio simple de los K vecinos más cercanos.", "La estandarización es importante porque las distancias dependen de la escala de las variables X.", res.get("uso_modo_pronostico", ""), res.get("implicacion_modo_pronostico", "")]
        )
        st.download_button("Descargar PDF", pdf_bytes, "informe_knn_regresion.pdf", "application/pdf")
    with c5:
        if st.button("Eliminar resultados guardados"):
            st.session_state.pop("resultados_knn_regresion", None)
            st.success("Resultados eliminados.")
