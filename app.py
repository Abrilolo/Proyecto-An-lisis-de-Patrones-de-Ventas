import streamlit as st
import pandas as pd

# --- IMPORTS LOCALES ---
from src.data_loader import cargar_y_limpiar_datos
from src.eda import (
    plot_tendencia_mensual, 
    plot_distribucion_categorias, 
    plot_heatmap_region_categoria, 
    plot_distribucion_pagos,
    plot_sankey_flujo, 
    plot_heatmap_semanal,
    get_tabla_top_diario,
    plot_valor_inventario_categoria, 
    plot_stock_vs_ventas,
    plot_ventas_por_region
)
from src.clustering import *
from src.clustering_viz import *
from src.forecasting import modelo_series_tiempo  # Usamos la función actualizada

# Configuración de página
st.set_page_config(page_title="Dashboard Comercial IA", layout="wide")

# --- 1. CARGA DE DATOS ---
with st.spinner('Cargando datos...'):
    df, df_stock = cargar_y_limpiar_datos()

if df is not None:
    # --- 2. LÓGICA ML (Clusters) ---
    if "Cluster_Label" not in df.columns:
        df = ejecutar_clustering(df)
        
    # Preparar lista de regiones
    lista_regiones = ["Todas"] + sorted(df["Region"].unique().tolist())

    # --- 3. DASHBOARD ---
    st.title("Análisis de Venta")
    st.markdown("---")

    # Pestañas
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Visión General", "🔎 EDA", "👤 Clustering", "📦 Inventario", "🔮 Predicción ML"])

    # === PESTAÑA 1: GENERAL ===
    with tab1:
        st.markdown("### Resumen de Métricas Clave")
        
        
        col_filtro_kpi, col_spacer = st.columns([1, 3])
        with col_filtro_kpi:
            region_kpi = st.selectbox("Filtrar KPIs por Región:", lista_regiones, key="sel_kpi_top")

        df_kpi = df if region_kpi == "Todas" else df[df["Region"] == region_kpi]

        kpi1, kpi2, kpi3 = st.columns(3)
        total_ingresos = df_kpi['Importe'].sum()
        kpi1.metric(f"Ingresos ({region_kpi})", f"${total_ingresos:,.0f}")
        
        total_transacciones = df_kpi['ID_Venta'].nunique()
        kpi2.metric(f"Transacciones ({region_kpi})", f"{total_transacciones}")
        
        promedio = df_kpi.groupby('ID_Venta')['Importe'].sum().mean() if total_transacciones > 0 else 0
        kpi3.metric(f"Ticket Promedio ({region_kpi})", f"${promedio:,.2f}")
        
        st.divider()
        
        # Ranking Regional
        st.subheader("Comparativa Regional")
        st.plotly_chart(plot_ventas_por_region(df), use_container_width=True, key="bar_region_tab1")
        
        st.divider()
        st.markdown("### Panorama Gráfico")
        
        c_trend_1, c_trend_2 = st.columns([1, 3])
        with c_trend_1:
            region_trend = st.selectbox("Región para Tendencia:", lista_regiones, key="sel_trend_tab1")
        
        df_trend = df if region_trend == "Todas" else df[df["Region"] == region_trend]
        
        st.subheader(f"Tendencia Mensual ({region_trend})")
        st.plotly_chart(plot_tendencia_mensual(df_trend), use_container_width=True, key="trend_tab1")
        
        st.divider()
        col_inf1, col_inf2 = st.columns(2)
        
        with col_inf1:
            region_pay = st.selectbox("Región para Pagos:", lista_regiones, key="sel_pay_tab1")
            df_pay = df if region_pay == "Todas" else df[df["Region"] == region_pay]
            st.subheader(f"Métodos de Pago ({region_pay})")
            st.plotly_chart(plot_distribucion_pagos(df_pay), use_container_width=True, key="pie_tab1")
            
        with col_inf2:
            region_cat = st.selectbox("Región para Categorías:", lista_regiones, key="sel_cat_tab1")
            df_cat = df if region_cat == "Todas" else df[df["Region"] == region_cat]
            st.subheader(f"Ventas por Categoría ({region_cat})")
            st.plotly_chart(plot_distribucion_categorias(df_cat), use_container_width=True, key="bar_tab1")

    # === PESTAÑA 2: EDA AVANZADO ===
    with tab2:
        st.header("Análisis de Patrones y Flujos")
        
        col_san_1, col_san_2, col_san_3 = st.columns([2, 1, 1])
        with col_san_1:
            st.subheader("Flujo de Ingresos")
            st.info("Filtra por **Categoría** para limpiar el gráfico.")
        with col_san_2:
            region_sankey = st.selectbox("Filtrar Región:", lista_regiones, key="sel_sankey_reg")
        with col_san_3:
            lista_categorias = ["Todas"] + sorted(df["Categoria"].unique().tolist())
            categoria_sankey = st.selectbox("Filtrar Categoría:", lista_categorias, key="sel_sankey_cat")
        
        df_sankey = df.copy()
        if region_sankey != "Todas":
            df_sankey = df_sankey[df_sankey["Region"] == region_sankey]
        if categoria_sankey != "Todas":
            df_sankey = df_sankey[df_sankey["Categoria"] == categoria_sankey]
        
        if not df_sankey.empty:
            st.plotly_chart(plot_sankey_flujo(df_sankey), use_container_width=True, key="sankey_tab2")
        else:
            st.warning("No hay datos para esta combinación.")
        
        st.markdown("---")
        
        # Heatmap
        col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
        with col_h1:
            st.subheader("Heatmap Semanal")
            st.write("Intensidad de ventas por día y semana.")
        with col_h2:
            region_heat = st.selectbox("Región:", lista_regiones, key="sel_heat_reg_2")
        with col_h3:
            cat_heat = st.selectbox("Categoría:", lista_categorias, key="sel_heat_cat_2")

        df_heat = df.copy()
        if region_heat != "Todas":
            df_heat = df_heat[df_heat["Region"] == region_heat]
        if cat_heat != "Todas":
            df_heat = df_heat[df_heat["Categoria"] == cat_heat]

        if not df_heat.empty:
            st.caption(f"Mostrando: **{cat_heat}** en **{region_heat}**")
            st.plotly_chart(plot_heatmap_semanal(df_heat), use_container_width=True, key="heat_week_tab2")
        else:
            st.warning("No hay datos.")
        
        st.markdown("---")
        st.subheader("Mapa Regional")
        st.plotly_chart(plot_heatmap_region_categoria(df), use_container_width=True, key="heat_region_tab2")
        

        st.markdown("---")
        st.subheader("Tabla de Productos Más Vendidos")
        
        c_f1, c_f2, c_f3 = st.columns(3)
        with c_f1:
            filtro_reg_tbl = st.selectbox("Región:", lista_regiones, key="tab2_tbl_reg")
        with c_f2:
            filtro_cat_tbl = st.selectbox("Categoría:", lista_categorias, key="tab2_tbl_cat")
        with c_f3:
            meses_ordenados = df.sort_values("Mes_Num")["Mes"].unique().tolist()
            filtro_mes_tbl = st.selectbox("Mes:", ["Todos"] + meses_ordenados, key="tab2_tbl_mes")

        df_tabla = df.copy()
        if filtro_reg_tbl != "Todas":
            df_tabla = df_tabla[df_tabla["Region"] == filtro_reg_tbl]
        if filtro_cat_tbl != "Todas":
            df_tabla = df_tabla[df_tabla["Categoria"] == filtro_cat_tbl]
        if filtro_mes_tbl != "Todos":
            df_tabla = df_tabla[df_tabla["Mes"] == filtro_mes_tbl]

        if not df_tabla.empty:
            top_global = df_tabla.groupby("Nombre_producto")["Importe"].sum().sort_values(ascending=False)
            if not top_global.empty:
                nom = top_global.index[0]
                val = top_global.iloc[0]
                k1, k2 = st.columns(2)
                k1.metric("Más vendido (Selección)", nom)
                k2.metric("Total Generado", f"${val:,.2f}")
            
            st.divider()
            st.write("#### Detalle diario:")
            st.dataframe(get_tabla_top_diario(df_tabla), use_container_width=True, hide_index=True)
        else:
            st.warning("No hay datos.")

# === PESTAÑA 3: SEGMENTACIÓN IA ===
    with tab3:
        st.header("Segmentación de Clientes")
        st.info("Este modelo agrupa a los **326 Clientes Únicos** basándose en su Frecuencia, Gasto y Recencia.")

        # 1. Generar Clusters
        # df_clientes tiene una fila por cliente. df_centroides son los puntos X del mapa.
        df_clientes, df_centroides = generar_clustering_clientes(df)

        # 2. Resumen Numérico (Tabla de Perfiles)
        st.subheader("Perfiles de los Clusters (Promedios)")
        perfil_cluster = df_clientes.groupby("Cluster")[
            ['Frecuencia_Compra', 'Importe_Total', 'Importe_Promedio', 'Recencia']
        ].mean().reset_index()
        
        st.dataframe(perfil_cluster.style.format({
            'Frecuencia_Compra': '{:.1f}',
            'Importe_Total': '${:,.2f}',
            'Importe_Promedio': '${:,.2f}',
            'Recencia': '{:.0f} días'
        }), use_container_width=True, hide_index=True)

        st.markdown("---")

        # 3. Distribución de Clientes
        total_clientes = df['ID_Cliente'].nunique()
        
        col_dist1, col_dist2 = st.columns([1, 2])
        with col_dist1:
            st.subheader("Clientes Totales")
            st.metric("A lo largo del tiempo", total_clientes)
            st.write("Distribución numérica:")
            distribucion = df_clientes["Cluster"].value_counts().sort_index()
            st.dataframe(distribucion, use_container_width=True)
            
        with col_dist2:
            st.subheader("Tamaño de los Grupos")
            st.plotly_chart(plot_distribucion_clusters(df_clientes), use_container_width=True)

        st.markdown("---") 

        # ==============================================================================
        # NUEVA SECCIÓN: EXPLORADOR DE INSIGHTS (CORREGIDA)
        # ==============================================================================
        st.header("Explorador de Insights por Cluster")
        st.write("Selecciona un grupo para ver qué compran y dónde viven.")

        # --- CORRECCIÓN DEL MERGE (Solución al KeyError) ---
        # 1. Aseguramos que los IDs sean texto (str) en ambos lados para que coincidan
        df['ID_Cliente'] = df['ID_Cliente'].astype(str)
        df_clientes['ID_Cliente'] = df_clientes['ID_Cliente'].astype(str)

        # 2. Realizamos el merge con 'suffixes'
        # Esto le dice a Pandas: "Si encuentras una columna repetida (como Cluster),
        # renombra la del df original a 'Cluster_trx' y deja la de clientes como 'Cluster'".
        # Así garantizamos que la columna "Cluster" exista limpia para los gráficos.
        df_transacciones_clusters = df.merge(
            df_clientes[['ID_Cliente', 'Cluster']], 
            on='ID_Cliente', 
            how='inner',
            suffixes=('_trx', '') 
        )
        # ---------------------------------------------------

        # Selector de Cluster
        lista_clusters = sorted(df_clientes["Cluster"].unique())
        cluster_seleccionado = st.selectbox(
            "Selecciona el Cluster a analizar:", 
            lista_clusters, 
            format_func=lambda x: f"Cluster {x}"
        )

        # Generar Gráficos de Insights
        c_insight_1, c_insight_2 = st.columns(2)
        
        with c_insight_1:
            st.caption(f"Top Productos del Cluster {cluster_seleccionado}")
            fig_prod = plot_top_productos_cluster(df_transacciones_clusters, cluster_seleccionado)
            st.plotly_chart(fig_prod, use_container_width=True)
            
        with c_insight_2:
            st.caption(f"Ubicación del Cluster {cluster_seleccionado}")
            fig_reg = plot_region_cluster(df_transacciones_clusters, cluster_seleccionado)
            st.plotly_chart(fig_reg, use_container_width=True)

        st.markdown("---")
        # ==============================================================================

        # 4. Visualización PCA
        st.subheader("Visualización PCA")
        st.write("Reducción de dimensionalidad para ver la separación de los grupos.")
        fig_pca = plot_pca_clientes(df_clientes, df_centroides)
        st.plotly_chart(fig_pca, use_container_width=True)

    # === PESTAÑA 4: INVENTARIO ===
    with tab4:
        st.header("Gestión de Inventario y Stock")
        
        lista_categorias_inv = ["Todas"] + sorted(df_stock["Categoria"].unique().tolist())
        cat_inv = st.selectbox("Filtrar por Categoría:", lista_categorias_inv, key="sel_inv_cat")
        
        if cat_inv != "Todas":
            df_stock_filtrado = df_stock[df_stock["Categoria"] == cat_inv]
            df_ventas_filtrada = df[df["Categoria"] == cat_inv]
        else:
            df_stock_filtrado = df_stock
            df_ventas_filtrada = df

        valor_stock = (df_stock_filtrado["Stock"] * df_stock_filtrado["Precio_Unitario"]).sum()
        items_stock = df_stock_filtrado["Stock"].sum()
        
        ki1, ki2 = st.columns(2)
        titulo_kpi = "Global" if cat_inv == "Todas" else cat_inv
        ki1.metric(f"Valor Stock ({titulo_kpi})", f"${valor_stock:,.0f}")
        ki2.metric(f"Unidades Físicas ({titulo_kpi})", f"{items_stock:,.0f}")
        
        st.markdown("---")
        st.subheader("Valor del Inventario")
        st.plotly_chart(plot_valor_inventario_categoria(df_stock_filtrado), use_container_width=True)
        
        st.markdown("---")
        st.subheader(f"Matriz de Rotación: {cat_inv}")
        if not df_stock_filtrado.empty:
            st.plotly_chart(plot_stock_vs_ventas(df_ventas_filtrada, df_stock_filtrado), use_container_width=True)
        else:
            st.warning("No hay productos.")

    # === PESTAÑA 5: LABORATOIO ML (SOLO SERIES DE TIEMPO) ===
    with tab5:
        st.header("Modelado Predictivo")
      
        st.subheader("Demanda Agregada (Series de Tiempo)")
        st.write("**Objetivo:** Predecir la demanda semanal futura utilizando el comportamiento histórico.")
        
        # Selector de Categoría
        cats = sorted(df["Categoria"].unique().tolist())
        cat_sel = st.selectbox("Categoría a pronosticar:", cats, key="sel_cat_lab")
        
        if st.button(f"Ejecutar Series de Tiempo para {cat_sel}", key="btn_run_lab"):
            with st.spinner(f"Generando modelo para {cat_sel}..."):
                # Ejecutamos la función del laboratorio
                res_ts, fig_ts = modelo_series_tiempo(df, cat_sel)
            
            # Diseño de resultados
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.write("#### 1. Validación (Métricas)")
                st.write("Comparamos nuestro modelo (Random Forest) contra un modelo ingenuo (Baseline).")
                # Usamos lightgreen para el fondo de las celdas mejores
                st.dataframe(res_ts.style.highlight_min(axis=0, subset=["RMSE", "MAE", "MAPE"], color="lightgreen"), use_container_width=True)
                st.caption("*Si el Random Forest tiene menor error que el Baseline, el modelo es útil.*")
                
            with c2:
                st.write("#### 2. Gráfico: Realidad vs Predicción")
                st.plotly_chart(fig_ts, use_container_width=True)

else:
    st.error("No se pudieron cargar los datos. Verifica tu conexión.")