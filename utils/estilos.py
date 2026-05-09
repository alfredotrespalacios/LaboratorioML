import streamlit as st

def mostrar_encabezado(titulo: str, descripcion: str = ""):
    st.title(titulo)
    if descripcion:
        st.write(descripcion)
    st.markdown("---")

def caja_pedagogica(texto: str):
    st.info(texto)

def advertencia_revision_resultados():
    st.warning(
        "Revise que los resultados cargados correspondan al análisis que desea incluir. "
        "Si no coinciden, elimine los resultados guardados y vuelva a ejecutar el módulo correspondiente."
    )
