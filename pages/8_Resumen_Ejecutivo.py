import streamlit as st

from utils.estilos import mostrar_encabezado, advertencia_revision_resultados
from utils.prompts import construir_prompt_integrado

st.set_page_config(page_title="Resumen ejecutivo", page_icon="📝", layout="wide")
mostrar_encabezado("📝 Resumen ejecutivo", "Construye un prompt integrado para solicitar a una IA la redacción del informe.")

mapa_resultados = {
    "Estadística descriptiva": "resultados_estadistica_descriptiva",
    "Regresión lineal": "resultados_regresion_lineal",
    "Otras regresiones": "resultados_otras_regresiones",
    "Clasificación": "resultados_clasificacion",
    "PCA": "resultados_pca",
    "Clustering": "resultados_clustering",
    "Red neuronal": "resultados_red_neuronal",
}

st.header("1. Gestión de resultados guardados")
claves_existentes = {nombre: clave for nombre, clave in mapa_resultados.items() if clave in st.session_state}

if claves_existentes:
    st.write("Resultados disponibles en esta sesión:")
    for nombre, clave in claves_existentes.items():
        resultado = st.session_state[clave]
        with st.expander(f"✅ {nombre}", expanded=False):
            st.write(f"**Archivo usado:** {resultado.get('archivo_usado', 'No disponible')}")
            st.write(f"**Fuente de datos:** {resultado.get('fuente_datos', 'No disponible')}")
            st.write(f"**Fecha de guardado:** {resultado.get('fecha_guardado', 'No disponible')}")
            st.write(f"**Tipo de análisis:** {resultado.get('tipo_analisis', nombre)}")
            if "descripcion_variables" in resultado:
                st.write("**Descripción de variables:**")
                st.json(resultado["descripcion_variables"])
            if st.button(f"Eliminar resultados de {nombre}", key=f"eliminar_{clave}"):
                del st.session_state[clave]
                st.success(f"Resultados de {nombre} eliminados.")
                st.rerun()
else:
    st.info("Todavía no hay resultados guardados en esta sesión.")

if st.button("Eliminar todos los resultados guardados"):
    for clave in mapa_resultados.values():
        st.session_state.pop(clave, None)
    st.success("Todos los resultados guardados fueron eliminados.")
    st.rerun()

st.markdown("---")
st.header("2. Seleccionar análisis para el informe")
analisis_seleccionados = st.multiselect("Seleccione uno o varios análisis que desea incluir", list(mapa_resultados.keys()), default=["Estadística descriptiva"])

resultados_cargados = {}
if analisis_seleccionados:
    st.subheader("Estado de carga")
    for analisis in analisis_seleccionados:
        clave = mapa_resultados[analisis]
        resultado = st.session_state.get(clave)
        if resultado is not None:
            resultados_cargados[analisis] = resultado
            st.success(f"{analisis}: resultados encontrados. Guardado: {resultado.get('fecha_guardado', 'Sin fecha')}")
        else:
            st.warning(f"{analisis}: todavía no hay resultados guardados. Primero ejecute el módulo correspondiente y presione 'Guardar resultados para resumen ejecutivo'.")

if resultados_cargados:
    st.subheader("Resultados que serán usados")
    advertencia_revision_resultados()
    for analisis, resultados in resultados_cargados.items():
        with st.expander(f"Ver resultados cargados: {analisis}", expanded=False):
            st.json(resultados)

st.markdown("---")
st.header("3. Contexto del informe")
objetivo = st.text_area("Objetivo general del informe", value="Redactar un informe ejecutivo que interprete los resultados cuantitativos disponibles.", height=90)
descripcion_datos = st.text_area("Descripción breve de los datos", value="Los datos corresponden a una base cuantitativa utilizada con fines académicos.", height=90)

with st.expander("Contexto noticioso: links web de noticias o referencias", expanded=True):
    st.write("Pegue hasta dos links de noticias o contexto. La app no lee estos links; los incorpora al prompt para orientar a la IA.")
    links_contexto = []
    for i in range(1, 3):
        links_contexto.append(st.text_input(f"Link de contexto noticioso {i}", key=f"link_contexto_{i}"))

with st.expander("Respuestas clave: preguntas que debe contestar el informe", expanded=True):
    st.write("Escriba hasta tres preguntas clave. El prompt pedirá responderlas en una sección llamada **Respuestas clave**.")
    preguntas_clave = []
    for i in range(1, 4):
        preguntas_clave.append(st.text_area(f"Pregunta clave {i}", key=f"pregunta_clave_{i}", height=70))

c1, c2 = st.columns(2)
with c1:
    audiencia = st.selectbox("Audiencia", ["Directivos", "Estudiantes", "Clientes", "Comité técnico", "Público general"])
    tono = st.selectbox("Tono deseado", ["Ejecutivo", "Académico", "Técnico", "Pedagógico", "Consultivo"])
with c2:
    extension = st.selectbox("Extensión deseada", ["Breve", "Media", "Amplia"], index=1)
    nivel_tecnico = st.selectbox("Nivel técnico", ["Bajo", "Medio", "Alto"], index=1)

st.markdown("---")
st.header("4. Generar prompt")
if not resultados_cargados:
    st.warning("No hay resultados cargados para construir el prompt.")
else:
    if st.button("Generar prompt con estos resultados"):
        prompt = construir_prompt_integrado(
            resultados_cargados,
            objetivo,
            descripcion_datos,
            audiencia,
            tono,
            extension,
            nivel_tecnico,
            links_contexto=links_contexto,
            preguntas_clave=preguntas_clave,
        )
        st.session_state["prompt_resumen_ejecutivo"] = prompt
        st.success("Prompt generado.")

if "prompt_resumen_ejecutivo" in st.session_state:
    st.subheader("Prompt generado")
    st.text_area("Copie este prompt y péguelo en una IA", value=st.session_state["prompt_resumen_ejecutivo"], height=500)
    st.download_button("Descargar prompt en TXT", data=st.session_state["prompt_resumen_ejecutivo"], file_name="prompt_resumen_ejecutivo.txt", mime="text/plain")

st.markdown("---")
st.caption("Esta página no llama directamente a una IA. Construye un prompt estructurado para que el estudiante lo copie y lo use en la herramienta de IA de su preferencia.")
