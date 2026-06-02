import streamlit as st

st.set_page_config(
    page_title="Laboratorio ML e IA para Finanzas",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Laboratorio de Machine Learning e Inteligencia Artificial para Finanzas")
st.subheader("Versión completa v1.16")

st.markdown(
    """
    **Aplicación creada por Alfredo Trespalacios** como complemento a las memorias del curso de
    **Machine Learning e Inteligencia Artificial para Finanzas**.
    """
)

st.info(
    """
    **Propósito pedagógico:** esta aplicación tiene únicamente fines educativos. Su objetivo es que los
    estudiantes comprendan conceptos de estadística, econometría básica y Machine Learning sin necesidad
    de programar. No se recomienda ni se autoriza su uso como herramienta para actividades profesionales,
    decisiones empresariales, decisiones financieras, consultoría, valoración, predicción operativa o
    toma de decisiones reales sin una validación técnica independiente.
    """
)

st.write(
    """
    Este laboratorio permite explorar análisis estadístico, econometría básica y modelos de Machine Learning
    desde una interfaz visual. La aplicación busca facilitar la comprensión conceptual, la interpretación de
    resultados y la conexión entre modelos cuantitativos e informes ejecutivos.
    """
)

st.markdown("---")

st.header("Flujo recomendado")

st.write(
    """
    1. Ingrese a un módulo de análisis.
    2. Use los datos de entrada por defecto o suba un Excel propio.
    3. Seleccione variables, describa su significado y configure el modelo.
    4. Ejecute el análisis.
    5. Revise tablas, pruebas, métricas y gráficos interactivos.
    6. Guarde los resultados para el resumen ejecutivo.
    7. Descargue los resultados en Excel, JSON o PDF si desea conservarlos.
    8. Use el módulo de resumen ejecutivo para construir un prompt integrado para IA.
    """
)

st.warning(
    """
    **Aclaración importante:** los resultados generados por esta aplicación deben interpretarse como ejercicios
    de aprendizaje. Los modelos, métricas, pruebas estadísticas y gráficos no sustituyen la revisión profesional,
    la validación metodológica ni el juicio experto requerido en aplicaciones reales.
    """
)

st.info(
    "Los archivos subidos por los estudiantes se usan temporalmente en memoria. "
    "No se guardan permanentemente en el servidor ni en el repositorio."
)

st.header("Módulos disponibles")

st.table(
    {
        "Módulo": [
            "1. Estadística descriptiva",
            "2. Regresión lineal",
            "3. KNN Regresión",
            "4. Árbol de decisión - Regresión",
            "5. Otras regresiones",
            "6. Clasificación",
            "7. Reducción de dimensionalidad - PCA",
            "8. Clustering",
            "9. Red neuronal",
            "10. Resumen ejecutivo",
        ],
        "Archivo de datos por defecto": [
            "data/datos_descriptiva.xlsx",
            "data/datos_regresion_lineal.xlsx",
            "data/datos_knn_regresion.xlsx",
            "data/datos_arbol_decision_regresion.xlsx",
            "data/datos_otras_regresiones.xlsx",
            "data/datos_clasificacion.xlsx",
            "data/datos_pca.xlsx",
            "data/datos_clustering.xlsx",
            "data/datos_red_neuronal.xlsx",
            "Usa resultados guardados",
        ],
        "Estado": ["Disponible"] * 10,
    }
)

st.markdown("---")

st.caption(
    "Use el menú lateral para navegar entre módulos. "
    "Cada módulo permite trabajar con datos por defecto o con un archivo Excel cargado por el usuario."
)
