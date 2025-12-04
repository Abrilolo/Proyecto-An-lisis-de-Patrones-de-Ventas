import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def generar_clustering_clientes(df):
    """
    Segmentación de Clientes (RFM + K-Means + PCA).
    Se usa en la pestaña 'Segmentación'.
    """
    # 1. Preparar datos (RFM)
    # Nota: df['Fecha'] ya es datetime desde el loader, no hace falta convertirla de nuevo.
    fecha_actual = df['Fecha'].max()

    # Agrupar por Cliente
    customer_data = df.groupby('ID_Cliente').agg(
        Frecuencia_Compra=('ID_Venta', 'nunique'),
        Importe_Total=('Importe', 'sum'),
        Importe_Promedio=('Importe', 'mean'),
        Ultima_Compra=('Fecha', 'max')
    ).reset_index()

    # Calcular Recencia (Días)
    customer_data['Recencia'] = (fecha_actual - customer_data['Ultima_Compra']).dt.days
    
    # Seleccionar features
    features = ['Frecuencia_Compra', 'Importe_Total', 'Importe_Promedio', 'Recencia']
    
    # 2. ML Pipeline (Escalado -> KMeans -> PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(customer_data[features])

    # K-Means (k=4)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    customer_data['Cluster'] = kmeans.fit_predict(X_scaled)

    # PCA para visualización
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X_scaled)
    
    customer_data['PCA1'] = pca_components[:, 0]
    customer_data['PCA2'] = pca_components[:, 1]

    # Calcular Centroides en PCA
    centroides_pca = pca.transform(kmeans.cluster_centers_)
    df_centroides = pd.DataFrame(centroides_pca, columns=['PCA1', 'PCA2'])
    df_centroides['Cluster'] = "Centroide"

    return customer_data, df_centroides

def ejecutar_clustering(df):
    """
    OPTIMIZADO: La segmentación de transacciones (fila por fila) no se estaba 
    utilizando en las visualizaciones principales (se usaba la de Clientes).
    
    Se mantiene una versión ligera para no romper la estructura de app.py,
    pero eliminamos el cálculo pesado de OneHotEncoding + KMeans sobre todo el dataset.
    Esto acelerará significativamente la carga.
    """
    # Asignamos un cluster genérico para cumplir con el requisito de estructura
    df["Cluster"] = 0
    df["Cluster_Label"] = "General"
    return df

def ejecutar_clustering_mensual(df):
    """
    Clasifica los meses en 'Regular' o 'Pico' según ventas y tráfico.
    Se usa en la pestaña 'Segmentación' (Gráfico de burbujas).
    """
    # 1. Preparar datos temporales
    df['Año'] = df['Fecha'].dt.year
    df['Mes_Num'] = df['Fecha'].dt.month

    # 2. Agregación Mensual
    monthly_data = df.groupby(['Año', 'Mes_Num']).agg(
        Volumen_Ventas=('Cantidad', 'sum'),
        Ingresos_Totales=('Importe', 'sum'),
        Numero_Transacciones=('ID_Venta', 'nunique'),
        Clientes_Unicos=('ID_Cliente', 'nunique')
    ).reset_index()

    # 3. K-Means (k=2: Alto vs Bajo)
    features = ['Volumen_Ventas', 'Ingresos_Totales', 'Numero_Transacciones', 'Clientes_Unicos']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(monthly_data[features])

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    monthly_data['Cluster_Mes'] = kmeans.fit_predict(X_scaled)

    # 4. Etiquetado Inteligente (Detectar cuál es el cluster "bueno")
    promedio_ventas = monthly_data.groupby("Cluster_Mes")["Ingresos_Totales"].mean().sort_values()
    
    mapa_etiquetas = {
        promedio_ventas.index[0]: "Regular",
        promedio_ventas.index[1]: "Pico / Alta"
    }
    monthly_data['Tipo_Mes'] = monthly_data['Cluster_Mes'].map(mapa_etiquetas)

    # Formateo para gráfica
    mapa_meses = {
        1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
    }
    monthly_data['Nombre_Mes'] = monthly_data['Mes_Num'].map(mapa_meses)
    monthly_data['Periodo'] = monthly_data['Año'].astype(str) + "-" + monthly_data['Nombre_Mes']

    return monthly_data