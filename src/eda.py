import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# --- FUNCIONES AUXILIARES ---

def hex_to_rgba(hex_color, opacity):
    """Convierte color Hex a RGBA para dar transparencia a los flujos."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({r},{g},{b},{opacity})'
    return hex_color

# --- GRÁFICOS BÁSICOS ---

def plot_tendencia_mensual(df):
    """Gráfico de línea: Tendencia de ventas por mes."""
    ventas_mes = df.groupby("AnioMes")["Importe"].sum().reset_index()
    ventas_mes = ventas_mes.sort_values("AnioMes")
    
    fig = px.line(
        ventas_mes, x="AnioMes", y="Importe", markers=True, 
        title="Tendencia Mensual de Ingresos"
    )
    fig.update_layout(xaxis_title="Mes", yaxis_title="Ventas ($)")
    return fig

def plot_distribucion_categorias(df):
    """Gráfico de barras: Ventas por categoría."""
    # Usamos "Categoria" sin tilde
    data = df.groupby("Categoria")["Importe"].sum().reset_index().sort_values("Importe", ascending=False)
    
    fig = px.bar(
        data, x="Categoria", y="Importe", color="Categoria", 
        title="Ranking de Ventas por Categoría"
    )
    return fig

def plot_heatmap_region_categoria(df):
    """Mapa de calor: Región vs Categoría."""
    pivot = df.pivot_table(index="Region", columns="Categoria", values="Importe", aggfunc="sum").fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="Viridis",
        text=pivot.values,
        texttemplate="%{text:.0f}"
    ))
    fig.update_layout(title="Mapa de Calor: Intensidad (Región vs Categoría)")
    return fig

def plot_distribucion_pagos(df):
    """Gráfico de pastel: Uso de métodos de pago (Estilizado)."""
    # 1. Agrupar y ORDENAR descendente
    data = df.groupby("Metodo")["Importe"].sum().reset_index()
    data = data.sort_values("Importe", ascending=False)
    
    # 2. Definir colores personalizados
    n_categorias = len(data)
    # Color destacado (Coral/Naranja) para el mayor valor
    color_destacado = "#A9BA48" 
    # Generamos azules invertidos (de oscuro a claro)
    colores_azules = px.colors.sequential.Blues[-n_categorias:] [::-1]
    # Combinamos: Destacado + Azules (saltando el primero azul para contraste)
    paleta_final = [color_destacado] + colores_azules[1:]

    fig = px.pie(
        data, 
        values="Importe", 
        names="Metodo", 
        title="Share de Métodos de Pago", 
        hole=0.4,
        color_discrete_sequence=paleta_final 
    )
    
    # 3. Ajustes visuales (Negritas y tamaño)
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label', 
        textfont_size=18,
        textfont_family="Arial Black", 
        textfont_color="white",
        marker=dict(line=dict(color='#000000', width=1))
    )
    
    fig.update_layout(height=600, font=dict(size=16))
    return fig

# --- GRÁFICOS AVANZADOS (Sankey & Heatmap Semanal) ---

def plot_sankey_flujo(df):
    """Diagrama de Sankey: Colores basados en el MÉTODO DE PAGO (Destino)."""
    
    # 1. Agrupar datos
    df_flow = df.groupby(["Categoria", "Metodo"])["Importe"].sum().reset_index()

    # 2. Listas de nodos
    categorias = df_flow["Categoria"].unique().tolist()
    metodos = df_flow["Metodo"].unique().tolist()
    all_labels = categorias + metodos
    
    # Mapa de índices
    label_map = {label: i for i, label in enumerate(all_labels)}
    
    # 3. Asignación de Colores para Métodos
    palette = px.colors.qualitative.Bold
    method_color_map = {m: palette[i % len(palette)] for i, m in enumerate(metodos)}
    
    # Construimos la lista de colores para TODOS los nodos
    # Categorías = Gris, Métodos = Color vibrante
    node_colors = []
    for label in all_labels:
        if label in categorias:
            node_colors.append("#636363")  # Gris neutro
        else:
            node_colors.append(method_color_map[label]) 

    # 4. Configurar links (conexiones)
    source_indices = [label_map[cat] for cat in df_flow["Categoria"]]
    target_indices = [label_map[met] for met in df_flow["Metodo"]]
    values = df_flow["Importe"].tolist()
    
    # Color del link = Color del Método destino (con transparencia)
    link_colors = [hex_to_rgba(method_color_map[m], 0.5) for m in df_flow["Metodo"]]

    # 5. Construir gráfico
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20, thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color=node_colors,
            hovertemplate='Total: $%{value:,.0f}<extra></extra>'
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color=link_colors,
            hovertemplate='%{source.label} → %{target.label}<br>Monto: $%{value:,.0f}<extra></extra>'
        )
    )])

    fig.update_layout(title_text="Flujo de Dinero: Categoría → Método de Pago", font_size=14)
    return fig

def plot_heatmap_semanal(df):
    """Mapa de Calor Temporal: Día de la semana vs Semana del año."""
    pivot = df.pivot_table(
        index="Semana", 
        columns="Dia_Semana", 
        values="Importe", 
        aggfunc="sum"
    ).fillna(0)

    dias_texto = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=dias_texto,
        y=pivot.index,
        colorscale="YlOrRd", # Escala de fuego (Amarillo a Rojo)
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Calendario de Ventas (Intensidad por Día y Semana)",
        xaxis_title="Día de la Semana",
        yaxis_title="Semana del Año"
    )
    return fig

def get_tabla_top_diario(df):
    """
    Genera una tabla con el producto más vendido de cada día.
    """
    if df.empty:
        return pd.DataFrame()

    # 1. Agrupar por Fecha y Producto, sumando el Importe y la Cantidad
    diario = df.groupby(["Fecha", "Nombre_producto"])[["Importe", "Cantidad"]].sum().reset_index()

    # 2. Ordenar: Primero por Fecha, luego por Importe (Descendente)
    diario = diario.sort_values(by=["Fecha", "Importe"], ascending=[True, False])

    # 3. Quedarse con el primer registro de cada fecha (el más vendido)
    top_diario = diario.drop_duplicates(subset="Fecha", keep="first")

    # 4. Limpieza para mostrar
    # Formateamos la fecha para que no salga la hora (00:00:00)
    top_diario["Fecha"] = top_diario["Fecha"].dt.strftime('%d-%m-%Y')
    
    # Renombrar columnas para que se vea profesional
    top_diario = top_diario.rename(columns={
        "Nombre_producto": "Producto Estrella",
        "Importe": "Venta Total ($)",
        "Cantidad": "Unidades"
    })

    return top_diario[["Fecha", "Producto Estrella", "Venta Total ($)", "Unidades"]]

def plot_valor_inventario_categoria(df_productos):
    """
    Muestra cuánto dinero tienes parado en almacén por categoría.
    Calcula: Stock * Precio Unitario.
    """
    # Calcular valor del stock
    # CORRECCIÓN: "Precio_Unitario" ya viene como float desde el loader
    df_productos["Valor_Stock"] = df_productos["Stock"] * df_productos["Precio_Unitario"]
    
    # CORRECCIÓN: Usar "Categoria" sin tilde
    data = df_productos.groupby("Categoria")["Valor_Stock"].sum().reset_index().sort_values("Valor_Stock", ascending=False)
    
    fig = px.bar(
        data, 
        x="Categoria", 
        y="Valor_Stock", 
        color="Categoria",
        title="Valor Monetario del Inventario por Categoría",
        labels={"Valor_Stock": "Valor en Inventario ($)"},
    )
    return fig

def plot_stock_vs_ventas(df_ventas, df_productos):
    """
    Gráfico de Dispersión: Stock vs. Demanda.
    """
    # 1. Calcular total vendido por producto
    ventas_por_prod = df_ventas.groupby("Nombre_producto")["Cantidad"].sum().reset_index()
    ventas_por_prod.rename(columns={"Cantidad": "Unidades_Vendidas"}, inplace=True)
    
    # 2. Unir con la tabla de productos
    df_mix = pd.merge(df_productos, ventas_por_prod, on="Nombre_producto", how="left").fillna(0)
    
    # 3. Crear gráfico
    fig = px.scatter(
        df_mix, 
        x="Unidades_Vendidas", 
        y="Stock", 
        color="Categoria", 
        size="Precio_Unitario", 
        hover_name="Nombre_producto",
        title="Matriz de Inventario: Stock vs. Demanda",
        text="Nombre_producto"
    )
    
    # Líneas promedio para dividir en cuadrantes (AHORA BLANCAS)
    prom_ventas = df_mix["Unidades_Vendidas"].mean()
    prom_stock = df_mix["Stock"].mean()
    
    # Línea Horizontal (Stock Promedio)
    fig.add_hline(
        y=prom_stock, 
        line_dash="dot", 
        line_color="white",  # <--- CAMBIO: Línea blanca
        annotation_text="Stock Promedio", 
        annotation_position="bottom right",
        annotation_font_color="white" # <--- CAMBIO: Texto blanco
    )
    
    # Línea Vertical (Ventas Promedio)
    fig.add_vline(
        x=prom_ventas, 
        line_dash="dot", 
        line_color="white",  # <--- CAMBIO: Línea blanca
        annotation_text="Ventas Promedio", 
        annotation_position="top right",
        annotation_font_color="white" # <--- CAMBIO: Texto blanco
    )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(height=600)
    
    return fig

def plot_ventas_por_region(df):
    """
    Gráfico de barras: Ranking de ventas por Región.
    Usa una paleta de azules.
    """
    # Agrupar por región y ordenar
    data = df.groupby("Region")["Importe"].sum().reset_index().sort_values("Importe", ascending=False)
    
    fig = px.bar(
        data, 
        x="Region", 
        y="Importe", 
        color="Importe", # El color depende del monto (más ventas = azul más fuerte)
        title="Ranking de Ventas por Región",
        color_continuous_scale="Blues", # Paleta Azul
        text_auto='.2s' # Formato compacto (ej: 1.2M, 500k)
    )
    
    # Ajustes de estilo
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(height=400, showlegend=False)
    
    return fig