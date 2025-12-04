# Dashboard Comercial con Análisis de Datos y Machine Learning

Este proyecto desarrolla un dashboard interactivo para el análisis integral de ventas, clientes e inventario a partir de datos transaccionales. La solución combina análisis exploratorio de datos (EDA), segmentación de clientes mediante clustering no supervisado y modelos de predicción de demanda basados en series de tiempo, integrados en una aplicación desarrollada con Streamlit.

El objetivo es transformar datos históricos en información accionable que apoye la toma de decisiones estratégicas y operativas en un contexto comercial.

---

## Estructura del Proyecto

```text
.
├── app.py
├── src/
│   ├── clustering.py
│   ├── clustering_viz.py
│   ├── data_loader.py
│   ├── eda.py
│   └── forecasting.py
├── requirements.txt
└── README.md



---

## Dependencias

Proyecto desarrollado en Python 3.10+.  
Principales librerías utilizadas:

- streamlit  
- pandas  
- numpy  
- scikit-learn  
- plotly  
- kagglehub  



---

## Flujo General del Proyecto

1. Carga e integración de datos transaccionales.
2. Limpieza y creación de variables temporales y monetarias.
3. Análisis exploratorio para identificación de patrones.
4. Segmentación de clientes mediante técnicas no supervisadas.
5. Modelado predictivo de demanda semanal.
6. Integración de resultados en un dashboard interactivo.

---

## Carga y Preparación de Datos

El módulo `src/data_loader.py` se encarga de:

- Descargar los datos desde Kaggle.
- Leer y unificar las fuentes de ventas, productos, clientes y métodos de pago.
- Convertir fechas a formato `datetime`.
- Crear variables derivadas como importe por venta y métricas temporales.
- Tratar valores nulos y registros inconsistentes.

El resultado es un DataFrame consolidado listo para análisis y modelado.

---

## Análisis Exploratorio de Datos (EDA)

En `src/eda.py` se implementan visualizaciones que permiten comprender la estructura y comportamiento de los datos. Entre ellas:

- Distribución de métodos de pago.
- Ventas totales por categoría.
- Comparativa de ventas por región.
- Mapas de calor región–categoría.
- Calendario de ventas por día y semana.
- Tendencia mensual de ingresos.

El EDA permite identificar patrones de consumo, concentración geográfica y estacionalidad antes de aplicar modelos más complejos.

---

## Segmentación de Clientes

La segmentación se realiza en `src/clustering.py` utilizando métricas RFM:

- Frecuencia de compra.
- Importe total y promedio.
- Recencia de la última compra.

Las variables son estandarizadas y se aplica el algoritmo K-Means con cuatro clusters. Posteriormente, se utiliza PCA para reducir la dimensionalidad y facilitar la interpretación visual de los segmentos.

El módulo `src/clustering_viz.py` complementa el análisis con:

- Distribución del tamaño de los clusters.
- Visualización de clientes en el espacio PCA.
- Identificación de perfiles promedio por grupo.

---

## Predicción de Demanda

El módulo `src/forecasting.py` implementa un modelo de series de tiempo a nivel semanal por categoría. El enfoque incluye:

- Agregación temporal de ventas.
- Creación de variables rezagadas (lags).
- Entrenamiento de un modelo Random Forest.
- Comparación contra un modelo base ingenuo.
- Evaluación con métricas RMSE, MAE y MAPE.

Los resultados permiten evaluar la utilidad del modelo para anticipar la demanda futura.

---

## Dashboard Interactivo

El archivo `app.py` integra todos los módulos anteriores en una aplicación Streamlit estructurada en:

- Visión general del negocio.
- Análisis exploratorio interactivo.
- Segmentación de clientes.
- Análisis de inventario.
- Predicción de ventas.

El dashboard permite aplicar filtros por categoría, región y tiempo, facilitando la exploración dinámica de los resultados.

---

## Cómo ejecutar el proyecto

1. Clona el repositorio:

```

https://github.com/Abrilolo/Proyecto-An-lisis-de-Patrones-de-Ventas.git


```

2. (Opcional) Crea un entorno virtual:

```

python -m venv .venv

```

3. Activa el entorno virtual:

**Windows**
```

.venv\Scripts\activate

```

**macOS / Linux**
```

source .venv/bin/activate

```

4. Instala las dependencias:

```

pip install -r requirements.txt

```

5. Ejecuta la aplicación del dashboard:

```

streamlit run app.py

```

6. Abre el enlace que Streamlit muestre en la terminal (generalmente `http://localhost:8501`) en tu navegador.




