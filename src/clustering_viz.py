import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_pca_clientes(df_clientes, df_centroides):
    """
    Genera el gráfico de dispersión 2D (PCA) de los clientes.
    Muestra los grupos por color y los centroides con una 'X' blanca.
    """
    # Convertir el número de cluster a string para que Plotly use colores discretos (categorías)
    # Ordenamos para que la leyenda salga bonita (0, 1, 2, 3)
    df_clientes = df_clientes.sort_values("Cluster")
    df_clientes['Cluster_Str'] = df_clientes['Cluster'].astype(str)

    # 1. Gráfico de dispersión de los clientes (Puntos)
    fig = px.scatter(
        df_clientes,
        x="PCA1",
        y="PCA2",
        color="Cluster_Str",
        title="Mapa de Segmentación de Clientes (PCA)",
        labels={"Cluster_Str": "Grupo"},
        # Usamos una paleta de colores profesional
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data=['ID_Cliente', 'Importe_Total', 'Frecuencia_Compra'] # Datos al pasar el mouse
    )

    # 2. Agregar los Centroides (Las "X" grandes)
    fig.add_trace(go.Scatter(
        x=df_centroides['PCA1'],
        y=df_centroides['PCA2'],
        mode='markers',
        marker=dict(
            symbol='x',
            size=15,
            color='white',  # <--- CAMBIO AQUÍ: De 'black' a 'white'
            line=dict(width=2)
        ),
        name='Centroides (Promedio)',
        showlegend=True
    ))

    # Ajustes finales de diseño
    fig.update_layout(
        xaxis_title="Componente Principal 1 (Volumen/Gasto)",
        yaxis_title="Componente Principal 2 (Temporalidad/Recencia)",
        height=600,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_distribucion_clusters(df_clientes):
    """
    Muestra cuántos clientes hay en cada cluster.
    """
    # 1. Contar clientes por cluster
    # Usamos value_counts para contar y reset_index para convertirlo en DataFrame
    data = df_clientes["Cluster"].value_counts().reset_index()
    data.columns = ["Cluster", "Cantidad_Clientes"]
    data = data.sort_values("Cluster") # Ordenar por número de cluster (0, 1, 2...)
    
    # Calcular porcentaje para el tooltip o texto
    total = data["Cantidad_Clientes"].sum()
    data["Porcentaje"] = (data["Cantidad_Clientes"] / total) * 100

    # Convertir cluster a string para que el eje X sea categórico (no numérico continuo)
    data["Cluster_Str"] = data["Cluster"].astype(str)

    # 2. Crear Gráfico de Barras
    fig = px.bar(
        data, 
        x="Cluster_Str", 
        y="Cantidad_Clientes",
        color_continuous_midpoint=0,
        title="Distribución de Clientes por Segmento",
        labels={"Cluster_Str": "Grupo (Cluster)", "Cantidad_Clientes": "Número de Clientes"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    # 3. Personalizar el texto (ej: "45 (15%)")
    fig.update_traces(
        texttemplate='%{y} (%{customdata:.1f}%)',
        customdata=data["Porcentaje"],
        textposition='outside'
    )
    
    fig.update_layout(height=600, showlegend=False)
    
    return fig

def plot_top_productos_cluster(df_merged, cluster_id):
    """
    Muestra los 5 productos más vendidos para un Cluster específico.
    """
    # Filtrar por el cluster seleccionado
    data = df_merged[df_merged["Cluster"] == cluster_id]
    
    # Agrupar por producto y sumar importe
    top_prods = data.groupby("Nombre_producto")["Importe"].sum().reset_index()
    top_prods = top_prods.sort_values("Importe", ascending=False).head(7) # Top 7
    
    fig = px.bar(
        top_prods, 
        x="Importe", 
        y="Nombre_producto", 
        orientation='h',
        title=f"Top Productos - Cluster {cluster_id}",
        labels={"Importe": "Ventas Totales ($)", "Nombre_producto": ""},
        color="Importe",
        color_continuous_scale="Teal"
    )
    
    fig.update_layout(yaxis=dict(autorange="reversed"), height=350) # Invertir orden para que el #1 salga arriba
    return fig

def plot_region_cluster(df_merged, cluster_id):
    """
    Muestra la distribución regional de un Cluster específico.
    """
    data = df_merged[df_merged["Cluster"] == cluster_id]
    
    # Contamos clientes únicos por región en este cluster
    region_counts = data.groupby("Region")["ID_Cliente"].nunique().reset_index()
    region_counts.columns = ["Region", "Clientes_Unicos"]
    region_counts = region_counts.sort_values("Clientes_Unicos", ascending=False)
    
    # 2. Definir colores personalizados
    n_categorias = len(data)
    # Color destacado (Coral/Naranja) para el mayor valor
    color_destacado = "#9C39FF" 
    # Generamos azules invertidos (de oscuro a claro)
    colores_azules = px.colors.sequential.Blues[-n_categorias:] [::-1]
    # Combinamos: Destacado + Azules (saltando el primero azul para contraste)
    paleta_final = [color_destacado] + colores_azules[1:]
    
    fig = px.pie(
        region_counts, 
        names="Region", 
        values="Clientes_Unicos", 
        title=f"Ubicación Geográfica - Cluster {cluster_id}",
        hole=0.4,
        color_discrete_sequence=paleta_final
    )
    fig.update_layout(height=350)
    return fig