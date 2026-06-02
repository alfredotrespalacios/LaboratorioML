# Laboratorio de Machine Learning e IA para Finanzas - versión completa v1.13

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

## Cambios v1.5

- Regresión lineal: reporte OLS reorganizado en tablas.
- Regresión lineal: ecuación de especificación antes de resultados.
- Regresión lineal: pronóstico con últimos N datos definidos en la columna izquierda.
- Regresión lineal: tabla y gráfico de pronóstico; gráfico de variable explicada histórica vs. estimada.
- Otras regresiones: gráfico de variable explicada real vs. predicha.
- Clasificación: en regresión logística se puede calcular P(Y=1|X), no solo clasificar.
- Informe ejecutivo: links web de contexto y hasta tres preguntas para sección “Respuestas clave”.
- Red neuronal: esquema visual de arquitectura y gráfico de variable objetivo real vs. resultado del modelo.
- Se incorpora descripción de variables en los módulos principales para Excel, PDF, JSON y prompt ejecutivo.


## Cambios v1.6

- Se verifica explícitamente que `utils/graficos.py` incluya:
  - `grafico_lineas_modelo`
  - `grafico_probabilidades`
  - `grafico_arquitectura_red`
- Se corrige el error de importación en Streamlit Cloud asociado a `grafico_lineas_modelo`.
- Se mantiene la versión con regresión lineal reorganizada, pronóstico, regresión logística como probabilidad, esquema visual de red neuronal y sección “Respuestas clave”.


## Cambios v1.7

- Se corrige la exportación a Excel para evitar avisos de recuperación al abrir archivos.
- El campo `reporte_statsmodels` ya no se escribe en la hoja `Metadatos`; se mantiene como hoja técnica `Reporte_OLS`.
- Se limpian caracteres no válidos para XML/Excel antes de escribir los libros.
- En el módulo de informe ejecutivo se dejan solo dos links de contexto noticioso.
- El prompt solicita una sección específica llamada “Contexto noticioso”.


## Cambios v1.8

- En el módulo de regresión lineal se agregan dos vistas:
  - Reporte visual organizado del modelo lineal.
  - Reporte completo de statsmodels en formato original.
- El reporte visual organiza resumen, coeficientes y diagnóstico de residuales en tablas limpias.
- Se mantiene el reporte OLS original como anexo técnico visible en la app y exportado al Excel en la hoja `Reporte_OLS`.


## Cambios v1.9

- En el módulo de red neuronal, cuando el problema es de regresión, se agrega una tabla comparativa de estadísticos descriptivos entre:
  - variable explicada histórica,
  - variable estimada por la red neuronal.
- La tabla incluye media, mediana, varianza, desviación estándar, mínimo, máximo, asimetría y percentiles 1, 5, 25, 50, 75, 95 y 99.
- La comparación descriptiva se exporta a Excel y al PDF del módulo.


## Cambios v1.10

- En el módulo de red neuronal para regresión se agregan residuales:
  - tabla con y real, y predicho y residual,
  - estadísticos descriptivos de residuales,
  - gráfica de residuales,
  - histograma de residuales.
- En el módulo de red neuronal para regresión se agrega pronóstico con los últimos N datos reservados:
  - parámetro en la columna izquierda,
  - tabla de pronóstico,
  - gráfica de calibración y pronóstico.
- Los nuevos resultados se exportan a Excel, PDF, JSON y se guardan para el informe ejecutivo.


## Cambios v1.11

- Se agrega el bloque común “Modo de pronóstico” en los módulos de regresión:
  - Regresión lineal.
  - Otras regresiones.
  - Red neuronal, cuando el tipo de problema es regresión.
- El usuario puede seleccionar:
  - Evaluar capacidad predictiva del modelo.
  - Generar pronóstico de valores no observados.
- Cada resultado guarda:
  - modo_pronostico,
  - uso_modo_pronostico,
  - implicacion_modo_pronostico.
- La explicación del modo aparece en pantalla, Excel, JSON, PDF y resultados para el informe ejecutivo.
- Cuando el modo es evaluación predictiva, se calculan métricas de pronóstico: MAE, MSE, RMSE y MAPE cuando aplica.
- Cuando el modo es valores no observados, el sistema pronostica filas donde Y está vacía y aclara que no hay error observable.


## Cambios v1.12

- En el módulo de red neuronal para regresión se asegura la visualización explícita de la tabla “Métricas de evaluación de pronóstico”.
- La tabla aparece cuando el modo seleccionado es “Evaluar capacidad predictiva del modelo”.
- Se añade texto pedagógico para interpretar MAE, RMSE y MAPE del pronóstico.
- La misma tabla queda disponible en Excel, PDF, JSON y resultados para el informe ejecutivo.


## Cambios v1.13

- Se agrega un módulo independiente `3_KNN_Regresion.py`.
- El nuevo módulo usa `data/datos_knn_regresion.xlsx`, copia pedagógica de los datos de regresión lineal para facilitar comparación.
- KNN Regresión permite seleccionar Y y X, transformaciones logarítmicas, K, estandarización, modo de pronóstico, error por K y ejemplo de vecinos más cercanos.
- Cuando se usa ln(Y), reporta resultados en logaritmo y también retransformados a nivel.
- Se elimina KNN Regressor de Otras regresiones para evitar duplicidad.
- Se agrega U de Theil al módulo de regresión lineal.
- El módulo Resumen Ejecutivo ahora incluye KNN Regresión.
- La app pasa a tener 9 módulos.
