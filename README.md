# Laboratorio de Machine Learning e IA para Finanzas - versión completa v1.4

Aplicación educativa en Streamlit con 8 módulos:

1. Estadística descriptiva
2. Regresión lineal
3. Otras regresiones
4. Clasificación
5. Reducción de dimensionalidad - PCA
6. Clustering
7. Red neuronal
8. Resumen ejecutivo

## Características

- Una sola app con páginas internas.
- Datos de entrada por defecto en Excel dentro de `data/`.
- Carga temporal de Excel propio del usuario.
- Gráficos interactivos con Plotly.
- Guardado temporal de resultados en `st.session_state`.
- Descarga de resultados en Excel y JSON.
- Eliminación de resultados guardados.
- Resumen ejecutivo integrado mediante prompt para IA.

## Cómo ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Nota de privacidad

Los archivos Excel subidos por el usuario se usan temporalmente en memoria y no se guardan permanentemente.


## Cambios pedagógicos v1.1

- Módulo de red neuronal: cuadro explicativo de parámetros de configuración.
- Módulo de red neuronal: explicación debajo de cada gráfica.
- Módulo de clustering: cuadro explicativo dinámico según el método seleccionado.
- Módulo de clustering: explicación debajo de cada gráfica.
- Módulo de otras regresiones: cuadro explicativo dinámico según el algoritmo seleccionado.
- Módulo de otras regresiones: cuadro explicativo sobre porcentaje de prueba y estandarización de variables X.
- Módulo de otras regresiones: explicación debajo de cada gráfica.
- Módulo de regresión lineal: conclusión explícita sobre normalidad de residuales con Jarque-Bera.


## Autoría y propósito

Aplicación creada por Alfredo Trespalacios como complemento a las memorias del curso de Machine Learning e Inteligencia Artificial para Finanzas.

Esta aplicación tiene únicamente fines pedagógicos. Su objetivo es que los estudiantes comprendan conceptos de estadística, econometría básica y Machine Learning sin necesidad de programar. No se recomienda ni se autoriza su uso como herramienta para actividades profesionales, decisiones empresariales, decisiones financieras, consultoría, valoración, predicción operativa o toma de decisiones reales sin una validación técnica independiente.

## Cambios v1.3

- Se agrega descarga de informe en PDF en los módulos 1 a 7.
- Los PDF incluyen metadatos, tablas principales, notas pedagógicas y gráficas estáticas exportadas desde Plotly.
- Los gráficos interactivos se mantienen en la app; en el PDF se guardan como imágenes estáticas.


## Cambios v1.4

- En los informes PDF, los valores numéricos de las tablas se muestran con máximo cuatro decimales.
- Los números muy pequeños o muy grandes se muestran en notación científica con cuatro decimales.
