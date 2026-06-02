import pandas as pd
import streamlit as st

def capturar_descripcion_variables(variables: list[str], key_prefix: str) -> dict:
    variables = list(dict.fromkeys([v for v in variables if v]))
    descripciones = {}
    with st.expander("Descripción de variables usadas en el análisis", expanded=True):
        st.write(
            "Explique brevemente qué significa cada variable. "
            "Esta información se guardará en los resultados y se usará en Excel, PDF, JSON y en el prompt del informe ejecutivo."
        )
        for var in variables:
            descripciones[var] = st.text_area(
                f"¿Qué significa la variable {var}?",
                value=st.session_state.get(f"{key_prefix}_desc_{var}", ""),
                key=f"{key_prefix}_desc_{var}",
                height=80,
            )
    return descripciones

def descripcion_variables_df(descripciones: dict) -> pd.DataFrame:
    if not descripciones:
        return pd.DataFrame(columns=["variable", "descripcion"])
    return pd.DataFrame([{"variable": k, "descripcion": v} for k, v in descripciones.items()])
