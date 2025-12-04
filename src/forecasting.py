import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import plotly.express as px
import plotly.graph_objects as go

# --- UTILS ---
def calcular_metricas(y_true, y_pred):
    """Calcula RMSE, MAE y MAPE."""
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred)
    }

def modelo_series_tiempo(df, categoria):
    """
    Agrupa por semana y usa lags para predecir el futuro.
    Gráfico: Entrenamiento (Azul) + Test Real (Azul fuerte) + Predicción (Rojo Punteado).
    """
    # 1. Agrupar
    df_cat = df[df["Categoria"] == categoria].copy()
    df_sem = df_cat.set_index("Fecha").resample('W-MON')["Cantidad"].sum().reset_index()
    
    # Lags
    df_sem["Lag_1"] = df_sem["Cantidad"].shift(1)
    df_sem["Lag_4"] = df_sem["Cantidad"].shift(4)
    df_sem.dropna(inplace=True)
    
    # 2. Split Temporal
    # Dejamos 4 semanas para test
    corte = len(df_sem) - 4
    train = df_sem.iloc[:corte]
    test = df_sem.iloc[corte:]
    
    X_train = train[["Lag_1", "Lag_4"]]
    y_train = train["Cantidad"]
    X_test = test[["Lag_1", "Lag_4"]]
    y_test = test["Cantidad"]
    
    # 3. Modelos
    pred_base = X_test["Lag_1"]
    
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    pred_model = model.predict(X_test)
    
    # 4. Resultados
    metrics_base = calcular_metricas(y_test, pred_base)
    metrics_model = calcular_metricas(y_test, pred_model)
    
    res_df = pd.DataFrame({
        "Modelo": ["Baseline (Naive)", "Random Forest (Lags)"],
        "RMSE": [metrics_base["RMSE"], metrics_model["RMSE"]],
        "MAE": [metrics_base["MAE"], metrics_model["MAE"]],
        "MAPE": [f"{metrics_base['MAPE']:.1%}", f"{metrics_model['MAPE']:.1%}"]
    })
    
    # --- GRÁFICO COMPLETO ---
    fig = go.Figure()
    
    # A. Historia (Entrenamiento) - Línea delgada
    fig.add_trace(go.Scatter(
        x=train["Fecha"], y=train["Cantidad"], 
        name="Entrenamiento",
        line=dict(color="#63C3FA", width=2)
    ))
    
    # Conector visual (para que no haya hueco entre train y test)
    fig.add_trace(go.Scatter(
        x=[train["Fecha"].iloc[-1], test["Fecha"].iloc[0]],
        y=[train["Cantidad"].iloc[-1], test["Cantidad"].iloc[0]],
        mode='lines', showlegend=False,
        line=dict(color='#63C3FA', width=2)
    ))

    # B. Realidad (Test) - Línea gruesa con puntos
    fig.add_trace(go.Scatter(
        x=test["Fecha"], y=test["Cantidad"], 
        name="Realidad (Test)",
        mode='lines+markers',
        line=dict(color="#F4D87A", width=4)
    ))
    
    # C. Predicción del Modelo - Punteada roja
    fig.add_trace(go.Scatter(
        x=test["Fecha"], y=pred_model, 
        name="Predicción Modelo", 
        mode='lines+markers',
        line=dict(color='#EF553B', width=3, dash='dot')
    ))
    
    fig.update_layout(
        title=f"Validación Temporal: {categoria}",
        xaxis_title="Semana",
        yaxis_title="Cantidad Vendida",
        hovermode="x unified"
    )
    
    return res_df, fig