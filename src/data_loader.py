import pandas as pd
import kagglehub
import os
import streamlit as st
import unicodedata

def normalize_columns(df):
    """Normaliza los nombres de las columnas."""
    new_cols = []
    for c in df.columns:
        c2 = c.strip()
        c2 = "".join(
            ch for ch in unicodedata.normalize("NFKD", c2)
            if not unicodedata.combining(ch)
        )
        c2 = c2.replace(" ", "_")
        new_cols.append(c2)
    df.columns = new_cols
    return df

def cap_outliers_iqr(df, col):
    """Recorta outliers usando IQR."""
    if not pd.api.types.is_numeric_dtype(df[col]):
        return df
        
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    df[col] = df[col].clip(lower=lower, upper=upper)
    return df

@st.cache_data
def cargar_y_limpiar_datos():
    """Pipeline completo de ETL."""
    
    # 1. DESCARGA
    print("Descargando datos de Kaggle...")
    try:
        path = kagglehub.dataset_download("dataregina/datasets-para-proyecto-bi")
    except Exception as e:
        st.error(f"Error descargando datos: {e}")
        return None, None

    # 2. CARGA
    try:
        df_ventas = pd.read_csv(os.path.join(path, "ventas.csv"))
        df_prod = pd.read_csv(os.path.join(path, "productos.csv"))
        df_cli = pd.read_csv(os.path.join(path, "clientes.csv"))
        df_metodos = pd.read_csv(os.path.join(path, "metodos_pago.csv"))
    except Exception as e:
        st.error(f"Error leyendo archivos: {e}")
        return None, None

    # 3. NORMALIZACIÓN
    df_ventas = normalize_columns(df_ventas)
    df_prod = normalize_columns(df_prod)
    df_cli = normalize_columns(df_cli)
    df_metodos = normalize_columns(df_metodos)

    # 4. LIMPIEZA
    if "Fecha_Resgistro" in df_cli.columns:
        df_cli.rename(columns={"Fecha_Resgistro": "Fecha_Registro"}, inplace=True)

    df_ventas["Fecha"] = pd.to_datetime(df_ventas["Fecha"], dayfirst=True, errors='coerce')
    
    if "Precio_Unitario" in df_prod.columns and df_prod["Precio_Unitario"].dtype == 'O':
        df_prod["Precio_Unitario"] = (
            df_prod["Precio_Unitario"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )

    # 5. NULOS Y DUPLICADOS
    cols_clave_ventas = ["ID_Venta", "Fecha", "ID_Cliente", "ID_Producto", "Cantidad", "Metodo_Pago"]
    cols_existentes = [c for c in cols_clave_ventas if c in df_ventas.columns]
    df_ventas.dropna(subset=cols_existentes, inplace=True)

    if "Precio_Unitario" in df_prod.columns:
        df_prod["Precio_Unitario"].fillna(df_prod["Precio_Unitario"].median(), inplace=True)
    
    if "Region" in df_cli.columns:
        df_cli["Region"].fillna("Sin_especificar", inplace=True)

    # El PDF elimina duplicados antes de calcular totales finales
    df_ventas.drop_duplicates(inplace=True)
    df_prod.drop_duplicates(inplace=True)
    df_cli.drop_duplicates(inplace=True)
    df_metodos.drop_duplicates(inplace=True)

    # --- 6. OUTLIERS (DESACTIVADO PARA COINCIDIR CON EL PDF) ---
    # Si activamos esto, los precios altos se recortan y el total de ventas baja.
    # Para tener los $103k exactos, debemos usar los precios originales.
    
    # if "Cantidad" in df_ventas.columns:
    #     df_ventas = cap_outliers_iqr(df_ventas, "Cantidad")
    # if "Precio_Unitario" in df_prod.columns:
    #     df_prod = cap_outliers_iqr(df_prod, "Precio_Unitario")
    # -----------------------------------------------------------

    # 7. MERGES
    df_ext = df_ventas.merge(df_prod[["ID_Producto", "Categoria", "Precio_Unitario", "Nombre_producto"]], 
                             on="ID_Producto", how="left")
    
    df_ext = df_ext.merge(df_cli[["ID_Cliente", "Region"]], on="ID_Cliente", how="left")
    
    left_key = "Metodo_Pago" if "Metodo_Pago" in df_ext.columns else "ID_Metodo"
    df_ext = df_ext.merge(df_metodos[["ID_Metodo", "Metodo"]], left_on=left_key, right_on="ID_Metodo", how="left")

    # 8. FEATURE ENGINEERING
    df_ext["Importe"] = df_ext["Cantidad"] * df_ext["Precio_Unitario"]
    
    df_ext["Ano"] = df_ext["Fecha"].dt.year
    df_ext["Dia"] = df_ext["Fecha"].dt.day
    df_ext["Semana"] = df_ext["Fecha"].dt.isocalendar().week
    df_ext["Dia_Semana"] = df_ext["Fecha"].dt.dayofweek
    df_ext["AnioMes"] = df_ext["Fecha"].dt.to_period("M").astype(str)

    # Mapa de Meses
    df_ext["Mes_Num"] = df_ext["Fecha"].dt.month
    mapa_meses = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }
    df_ext["Mes"] = df_ext["Mes_Num"].map(mapa_meses)

    df_ext.dropna(subset=["Importe"], inplace=True)
    
    return df_ext, df_prod