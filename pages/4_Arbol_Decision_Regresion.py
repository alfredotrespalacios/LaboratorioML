from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

from utils.carga_datos import cargar_datos_modulo, detectar_columnas_fecha, detectar_columnas_numericas
from utils.exportacion import preparar_excel_generico, preparar_json_descarga
from utils.graficos import grafico_histograma_interactivo, grafico_importancia_variables, grafico_lineas_modelo, grafico_real_vs_predicho, grafico_residuales
from utils.pdf_reportes import crear_pdf_modulo
from utils.pronostico import MODO_EVALUACION, MODO_VALORES_NO_OBSERVADOS, agregar_errores_pronostico, descripcion_modo_pronostico
from utils.regresion_helpers import metricas_pronostico_ext, metricas_regresion_ext, preparar_tabla_doble_escala, tabla_interpretacion_error_medio, tabla_modo_pronostico
from utils.transformaciones import aplicar_log_si_corresponde, ordenar_por_fecha
from utils.variables import capturar_descripcion_variables, descripcion_variables_df
from utils.estilos import mostrar_encabezado, caja_pedagogica

st.set_page_config(page_title="Árbol de decisión - Regresión", page_icon="🌳", layout="wide")
mostrar_encabezado("🌳 Árbol de decisión - Regresión", "Modelo basado en reglas para predecir una variable continua.")

caja_pedagogica(
    "Un árbol de decisión divide los datos en grupos mediante reglas sucesivas sobre las variables X. "
    "En cada hoja, el pronóstico corresponde al promedio de Y de las observaciones que caen en esa hoja."
)

with st.expander("Explicación pedagógica del método", expanded=True):
    st.markdown(
        """
        **Árbol de decisión para regresión:** construye reglas del tipo `si X <= umbral`, dividiendo la muestra en grupos cada vez más homogéneos en la variable objetivo.

        **Profundidad máxima:** controla cuántos niveles puede tener el árbol. Una profundidad mayor permite reglas más detalladas, pero aumenta el riesgo de sobreajuste.

        **Mínimo de observaciones por hoja:** exige que cada grupo final tenga un número mínimo de datos. Valores mayores producen árboles más simples y estables.

        **Mínimo de observaciones para dividir un nodo:** define cuántos datos debe tener un grupo antes de poder dividirse de nuevo.

        **Criterio fijo:** `squared_error`. El árbol busca divisiones que reduzcan el error cuadrático medio dentro de los grupos.

        **random_state fijo:** `42`. Sirve para que los resultados sean reproducibles en clase.
        """
    )

df, nombre_archivo, fuente_datos = cargar_datos_modulo("data/datos_arbol_decision_regresion.xlsx", "arbol_decision_regresion")
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

st.sidebar.header("4. Transformaciones")
usar_log_y = st.sidebar.radio(f"Transformación de {y_col}", ["Nivel", "ln(Y)"], horizontal=True) == "ln(Y)"
transformaciones_x = {}
for x in x_cols:
    transformaciones_x[x] = st.sidebar.radio(f"Transformación de {x}", ["Nivel", f"ln({x})"], horizontal=True, key=f"tree_log_{x}") != "Nivel"

st.sidebar.header("5. Parámetros del árbol")
max_depth = st.sidebar.slider("Profundidad máxima del árbol", 1, 20, 4)
min_samples_leaf = st.sidebar.slider("Mínimo de observaciones por hoja", 1, 30, 5)
min_samples_split = st.sidebar.slider("Mínimo de observaciones para dividir un nodo", 2, 50, 10)
profundidad_visual = st.sidebar.slider("Profundidad máxima a mostrar en la figura", 1, 5, 3)

descripcion_variables = capturar_descripcion_variables([y_col] + x_cols, "arbol_decision_regresion")

if "resultados_arbol_decision_regresion_actuales" not in st.session_state:
    st.session_state["resultados_arbol_decision_regresion_actuales"] = None

def preparar_datos(df_base):
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
    return datos.replace([np.inf, -np.inf], np.nan), y_nombre, x_nombres

def extraer_reglas(modelo, feature_names):
    """
    Extrae reglas completas desde la raíz hasta cada hoja del árbol.

    Cada fila corresponde a una hoja. La regla muestra las condiciones sucesivas
    que debe cumplir una observación para llegar a esa hoja.
    """
    tree = modelo.tree_
    feature_names = list(feature_names)
    reglas = []

    def recorrer(nodo, condiciones):
        izquierda = tree.children_left[nodo]
        derecha = tree.children_right[nodo]

        if izquierda == derecha:
            valor_estimado = float(tree.value[nodo][0][0])
            n_obs = int(tree.n_node_samples[nodo])

            if condiciones:
                regla_texto = "Si " + " y ".join(condiciones)
            else:
                regla_texto = "Todas las observaciones"

            reglas.append({
                "regla_id": len(reglas) + 1,
                "regla_texto": regla_texto,
                "valor_estimado": valor_estimado,
                "n_observaciones_en_hoja": n_obs,
            })
            return

        feature_idx = tree.feature[nodo]
        threshold = float(tree.threshold[nodo])
        nombre_variable = feature_names[feature_idx]

        recorrer(izquierda, condiciones + [f"{nombre_variable} <= {threshold:.4f}"])
        recorrer(derecha, condiciones + [f"{nombre_variable} > {threshold:.4f}"])

    recorrer(0, [])
    return pd.DataFrame(reglas)

if st.button("Entrenar árbol de decisión"):
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

    X_train = train_df[x_nombres]
    y_train = train_df[y_nombre]

    modelo = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=42,
    )
    modelo.fit(X_train, y_train)
    y_pred_train = modelo.predict(X_train)

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
        y_fore = modelo.predict(fore_df[x_nombres])
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

    importancia_df = pd.DataFrame({"variable": x_nombres, "importancia": modelo.feature_importances_}).sort_values("importancia", ascending=False)
    reglas_df = extraer_reglas(modelo, x_nombres)
    reglas_mostrar_df = reglas_df.head(10)

    st.session_state["resultados_arbol_decision_regresion_actuales"] = {
        "tipo_analisis": "Árbol de decisión - Regresión",
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
        "criterio": "squared_error",
        "random_state": 42,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "min_samples_split": min_samples_split,
        "profundidad_visual": profundidad_visual,
        **info_modo,
        "n_pronostico": int(n_pronostico) if modo_pronostico == MODO_EVALUACION else None,
        "metricas": metricas,
        "metricas_nivel": metricas_nivel,
        "metricas_pronostico": metricas_fore,
        "metricas_pronostico_nivel": metricas_fore_nivel,
        "predicciones": pred_df.to_dict(orient="records"),
        "pronostico": pronostico_df.to_dict(orient="records"),
        "serie_modelo": serie_modelo_df.to_dict(orient="records"),
        "importancia_variables": importancia_df.to_dict(orient="records"),
        "reglas": reglas_df.to_dict(orient="records"),
        "_x_nombres": x_nombres,
        "_modelo": modelo,
    }

res = st.session_state["resultados_arbol_decision_regresion_actuales"]
if res:
    st.subheader("Modo de pronóstico usado")
    st.info(f"**{res.get('modo_pronostico')}**\n\n**Uso:** {res.get('uso_modo_pronostico')}\n\n**Implicación:** {res.get('implicacion_modo_pronostico')}")

    pred_df = pd.DataFrame(res["predicciones"])
    pronostico_df = pd.DataFrame(res.get("pronostico", []))
    serie_modelo_df = pd.DataFrame(res.get("serie_modelo", []))
    importancia_df = pd.DataFrame(res.get("importancia_variables", []))
    reglas_df = pd.DataFrame(res.get("reglas", []))
    reglas_mostrar_df = reglas_df.head(10)
    desc_df = descripcion_variables_df(res.get("descripcion_variables", {}))
    metricas_df = pd.DataFrame([res["metricas"]])
    metricas_nivel_df = pd.DataFrame([res.get("metricas_nivel", {})])
    metricas_pronostico_df = pd.DataFrame([res.get("metricas_pronostico", {})])
    metricas_pronostico_nivel_df = pd.DataFrame([res.get("metricas_pronostico_nivel", {})])

    st.subheader("Configuración del árbol")
    st.dataframe(pd.DataFrame([{
        "criterio": res.get("criterio"),
        "random_state": res.get("random_state"),
        "max_depth": res.get("max_depth"),
        "min_samples_leaf": res.get("min_samples_leaf"),
        "min_samples_split": res.get("min_samples_split"),
        "profundidad_visual": res.get("profundidad_visual"),
    }]), use_container_width=True)

    st.subheader("Figura del árbol")
    st.caption(f"La figura muestra máximo {res.get('profundidad_visual')} niveles, aunque el árbol estimado pueda tener más.")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(18, 8))
        plot_tree(res["_modelo"], feature_names=res["_x_nombres"], filled=True, rounded=True, max_depth=res.get("profundidad_visual"), fontsize=8, ax=ax)
        st.pyplot(fig)
    except Exception as exc:
        st.warning(f"No fue posible mostrar la figura del árbol: {exc}")

    with st.expander("Reglas principales del árbol", expanded=True):
        st.write(
            "Cada fila corresponde a una hoja del árbol. La columna `regla_texto` muestra el camino de condiciones "
            "desde la raíz hasta esa hoja, y `valor_estimado` es el promedio de Y que el árbol asigna a las observaciones "
            "que cumplen esa regla."
        )
        st.write("Se muestran máximo 10 reglas principales en pantalla. Todas las reglas disponibles se exportan a Excel, PDF, JSON y resultados para el informe ejecutivo.")
        columnas_reglas = [c for c in ["regla_id", "regla_texto", "valor_estimado", "n_observaciones_en_hoja"] if c in reglas_mostrar_df.columns]
        st.dataframe(reglas_mostrar_df[columnas_reglas], use_container_width=True)

    st.subheader("Métricas de calibración")
    st.dataframe(metricas_df, use_container_width=True)
    if res.get("metricas_nivel"):
        st.write("**Métricas en nivel original de Y**")
        st.dataframe(metricas_nivel_df, use_container_width=True)

    st.subheader("Variable explicada real vs. predicha")
    st.plotly_chart(grafico_lineas_modelo(pred_df, "eje_x", ["y_real", "y_predicho"], "Árbol: variable explicada real vs. predicha"), use_container_width=True)
    st.plotly_chart(grafico_real_vs_predicho(pred_df["y_real"], pred_df["y_predicho"]), use_container_width=True)

    st.subheader("Residuales")
    st.dataframe(pred_df, use_container_width=True)
    st.plotly_chart(grafico_residuales(pred_df["residual"], "Residuales del árbol"), use_container_width=True)
    st.plotly_chart(grafico_histograma_interactivo(pred_df["residual"], "Distribución de residuales del árbol"), use_container_width=True)

    st.subheader("Pronóstico")
    if pronostico_df.empty:
        st.info("No hay filas disponibles para pronóstico según el modo seleccionado.")
    else:
        st.dataframe(pronostico_df, use_container_width=True)
        if res.get("metricas_pronostico"):
            st.write("**Métricas de evaluación de pronóstico**")
            st.dataframe(metricas_pronostico_df, use_container_width=True)
            if "Error_medio_pronostico" in metricas_pronostico_df.columns:
                st.dataframe(tabla_interpretacion_error_medio(metricas_pronostico_df.loc[0, "Error_medio_pronostico"], "Error medio de pronóstico"), use_container_width=True)
        if res.get("metricas_pronostico_nivel"):
            st.write("**Métricas de evaluación de pronóstico en nivel original de Y**")
            st.dataframe(metricas_pronostico_nivel_df, use_container_width=True)
        st.plotly_chart(grafico_lineas_modelo(serie_modelo_df, "eje_x", ["y_observada", "y_estimada", "y_pronosticada"], "Árbol: calibración y pronóstico"), use_container_width=True)

    st.subheader("Importancia de variables")
    st.plotly_chart(grafico_importancia_variables(importancia_df), use_container_width=True)

    if not desc_df.empty:
        st.subheader("Descripción de variables")
        st.dataframe(desc_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Gestión de resultados")
    c1, c2, c3, c4, c5 = st.columns(5)
    clean_res = {k:v for k,v in res.items() if not k.startswith("_")}
    with c1:
        if st.button("Guardar resultados para resumen ejecutivo"):
            st.session_state["resultados_arbol_decision_regresion"] = {**clean_res, "fecha_guardado": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            st.success("Resultados guardados.")
    with c2:
        excel = preparar_excel_generico(clean_res, {
            "Descripcion_variables": desc_df,
            "Configuracion": pd.DataFrame([{
                "criterio": res.get("criterio"),
                "random_state": res.get("random_state"),
                "max_depth": res.get("max_depth"),
                "min_samples_leaf": res.get("min_samples_leaf"),
                "min_samples_split": res.get("min_samples_split"),
            }]),
            "Modo_pronostico": tabla_modo_pronostico(res),
            "Metricas": metricas_df,
            "Metricas_nivel": metricas_nivel_df,
            "Metricas_pronostico": metricas_pronostico_df,
            "Metricas_pronostico_nivel": metricas_pronostico_nivel_df,
            "Predicciones": pred_df,
            "Pronostico": pronostico_df,
            "Serie_modelo": serie_modelo_df,
            "Importancia_variables": importancia_df,
            "Reglas": reglas_df,
        })
        st.download_button("Descargar Excel", excel, "resultados_arbol_decision_regresion.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c3:
        st.download_button("Descargar JSON", preparar_json_descarga(clean_res), "resultados_arbol_decision_regresion.json", "application/json")
    with c4:
        figs = [
            ("Variable explicada real vs. predicha", grafico_lineas_modelo(pred_df, "eje_x", ["y_real", "y_predicho"], "Árbol: variable explicada real vs. predicha"), "Compara la variable real con la predicción del árbol en calibración."),
            ("Residuales del árbol", grafico_residuales(pred_df["residual"], "Residuales del árbol"), "Muestra los errores de calibración del árbol."),
            ("Calibración y pronóstico", grafico_lineas_modelo(serie_modelo_df, "eje_x", ["y_observada", "y_estimada", "y_pronosticada"], "Árbol: calibración y pronóstico"), "Muestra calibración y pronóstico según el modo seleccionado."),
            ("Importancia de variables", grafico_importancia_variables(importancia_df), "Indica qué variables fueron más usadas para reducir error en las divisiones."),
        ]
        pdf_bytes = crear_pdf_modulo(
            "Informe del módulo: Árbol de decisión - Regresión",
            clean_res,
            tablas=[
                ("Descripción de variables", desc_df),
                ("Configuración", pd.DataFrame([{
                    "criterio": res.get("criterio"),
                    "random_state": res.get("random_state"),
                    "max_depth": res.get("max_depth"),
                    "min_samples_leaf": res.get("min_samples_leaf"),
                    "min_samples_split": res.get("min_samples_split"),
                }])),
                ("Modo de pronóstico", tabla_modo_pronostico(res)),
                ("Métricas", metricas_df),
                ("Métricas de pronóstico", metricas_pronostico_df),
                ("Predicciones", pred_df),
                ("Pronóstico", pronostico_df),
                ("Importancia de variables", importancia_df),
                ("Reglas", reglas_df),
            ],
            figuras=figs,
            notas=["El árbol divide los datos mediante reglas sucesivas. La tabla de reglas muestra el camino completo hasta cada hoja y el valor estimado de Y.", "La figura del árbol se limita visualmente para mantener legibilidad.", res.get("uso_modo_pronostico", ""), res.get("implicacion_modo_pronostico", "")]
        )
        st.download_button("Descargar PDF", pdf_bytes, "informe_arbol_decision_regresion.pdf", "application/pdf")
    with c5:
        if st.button("Eliminar resultados guardados"):
            st.session_state.pop("resultados_arbol_decision_regresion", None)
            st.success("Resultados eliminados.")
