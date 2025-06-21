import streamlit as st
import pandas as pd
from collections import OrderedDict
from pulp import *
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import io

# Configuración general
sns.set_theme(style="ticks")
st.set_page_config(page_title="Modelo de Optimización de Carbón", layout="wide")
st.title("🧮 Modelo de Optimización de Compras de Carbón")
st.markdown("Minimiza el costo total de compra de carbón cumpliendo restricciones de calidad, mezcla y disponibilidad.")

# Cargar archivo
archivo = st.file_uploader("📤 Carga el archivo de datos (.xlsx):", type=["xlsx"])

if archivo:
    hoja = pd.read_excel(archivo, sheet_name=0, header=None)

    # === Preprocesamiento ===
    df = hoja.iloc[2:, 0:9].copy()
    df.columns = hoja.iloc[1, 0:9]
    df = df[df['Disponible'].notnull() & df['Precio'].notnull()]
    df[['Disponible', 'Precio', 'HT', 'CZ', 'MV', 'S', 'FSI']] = df[
        ['Disponible', 'Precio', 'HT', 'CZ', 'MV', 'S', 'FSI']
    ].apply(pd.to_numeric, errors='coerce')
    df = df[df['Disponible'] > 0]
    df = df.drop_duplicates(subset=['Proveedor', 'Tipo']).sort_values(['Proveedor', 'Tipo'])
    df['Costo_CCB'] = df['Precio']/((1-df['MV'])/(1-0.012))

    # Parámetros
    requerimiento_total = float(hoja.iloc[2, 10])
    calidad_esperada = hoja.iloc[2, 12:16]
    calidad_esperada.index = hoja.iloc[1, 12:16]
    calidad_esperada_dict = calidad_esperada.to_dict()

    limites = hoja.iloc[2:, 17:19].dropna()
    limites.columns = hoja.iloc[1, 17:19]
    limites_dict = dict(zip(limites['TIPO'], limites['LIMITE']))

    # Diccionarios necesarios
    pares = list(OrderedDict.fromkeys(zip(df['Proveedor'], df['Tipo'])))
    atributos = ['Costo_CCB','Precio', 'Disponible', 'HT', 'CZ', 'MV', 'S', 'FSI']
    datos = {atr: {(p, t): row[atr] for p, t, row in zip(df['Proveedor'], df['Tipo'], df.to_dict('records'))} for atr in atributos}
    tipos = df['Tipo'].unique().tolist()

    # === Definición del modelo ===
    model = LpProblem("Optimizacion_Compras_Carbon", LpMinimize)
    x = LpVariable.dicts("Pedido", pares, lowBound=0, cat='Continuous')

    # Objetivo
    model += lpSum(x[par] * datos['Costo_CCB'][par] for par in pares)

    # Restricciones de cantidad total y disponibilidad
    model += lpSum(x[par] for par in pares) == requerimiento_total, "Requerimiento_Total"
    for par in pares:
        model += x[par] <= datos['Disponible'][par], f"Disponible_{par}"

    # Restricciones por tipo
    for tipo in tipos:
        limite = limites_dict.get(tipo, 1)
        model += lpSum(x[par] for par in pares if par[1] == tipo) <= limite * requerimiento_total, f"Max_{tipo}"

    # Restricciones de calidad
    total_pedido = lpSum(x[par] for par in pares)
    model += lpSum(x[par] * datos['S'][par] for par in pares) <= calidad_esperada_dict['S'] * total_pedido, "S"
    model += lpSum(x[par] * datos['FSI'][par] for par in pares) >= calidad_esperada_dict['FSI'] * total_pedido, "FSI"
    model += lpSum(x[par] * datos['CZ'][par] for par in pares) <= calidad_esperada_dict['CZ'] * total_pedido, "CZ"
    model += lpSum(x[par] * datos['MV'][par] for par in pares) <= calidad_esperada_dict['MV'] * total_pedido, "MV"

    # === Resolución del modelo ===
    with st.spinner("🔄 Ejecutando modelo..."):
        model.solve()

    st.success("✅ Modelo resuelto")
    st.write(f"**Estado:** {LpStatus[model.status]}")

    # === Resultados ===
    solucion = {par: x[par].varValue for par in pares if x[par].varValue > 0}
    df_sol = pd.DataFrame([
        {"Proveedor": p, "Tipo": t, "Toneladas": val}
        for (p, t), val in solucion.items()
    ])

    st.write(f"**RESULTADOS DEL MODELO:**")
    st.dataframe(df_sol, use_container_width=True)

    # Calidad alcanzada
    total = sum(solucion.values())
    if total > 0:
        s_prom = sum(datos['S'][par] * cantidad for par, cantidad in solucion.items()) / total
        fsi_prom = sum(datos['FSI'][par] * cantidad for par, cantidad in solucion.items()) / total
        cz_prom = sum(datos['CZ'][par] * cantidad for par, cantidad in solucion.items()) / total
        mv_prom = sum(datos['MV'][par] * cantidad for par, cantidad in solucion.items()) / total
        st.write(f"**Calidad Alcanzada:** S: {s_prom * 100:.2f}%, FSI: {fsi_prom:.2f}, CZ: {cz_prom * 100:.2f}%, MV: {mv_prom * 100:.2f}%")
    
    # Costo Total
    # Unir los DataFrames por el campo "Proveedor"
    df_resultado = pd.merge(df_sol, df[['Proveedor', 'Precio']], on='Proveedor', how='left')
    df_resultado['Total'] = df_resultado['Toneladas'] * df_resultado['Precio']
    costo_total = df_resultado['Total'].sum()
    
    # Coque bruto Producido
    rendimiento = ((1-mv_prom )/(1-0.012))
    coque_bruto_producido = total * rendimiento

    # Costo unitario del coque bruto producido
    costo_unitario_cbp = costo_total/coque_bruto_producido

    # Mostrar resultado en un dataframe
    st.write(f"**CALIDAD Y RENDIMIENTO ALCANZADO:**")
    # Crear un diccionario con los resultados formateados
    resumen = {
        'S (%)': f"{s_prom * 100:.2f}",
        'FSI': f"{fsi_prom:.2f}",
        'CZ (%)': f"{cz_prom * 100:.2f}",
        'MV (%)': f"{mv_prom * 100:.2f}",
        'Costo Total ($)': f"${costo_total:,.2f}",
        'Rendimiento Coque Bruto (%)': f"{rendimiento * 100:.2f}",
        'Total Coque Bruto Producido': f"{coque_bruto_producido:,.2f}",
        'Costo Unitario Coque Bruto ($/t)': f"${costo_unitario_cbp:,.2f}"
    }

    # Convertir a DataFrame con una sola fila
    df_resumen = pd.DataFrame([resumen])
    st.dataframe(df_resumen, use_container_width=True)


    # === Exportar Excel ===
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df_sol.to_excel(writer, index=False, sheet_name='Solución')
        excel_buffer.seek(0)

    if not df_sol.empty:
        col1, col2, col3 = st.columns([5, 1, 1])
        with col1:
            st.download_button(
                label="📥 Descargar Excel",
                data=excel_buffer,
                file_name="solucion_optima.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("No hay datos para exportar.")

    # === Gráfico de Torta por Tipo ===
    tipo_cantidad = df_sol.groupby("Tipo")["Toneladas"].sum()
    colors = sns.color_palette("colorblind", n_colors=len(tipo_cantidad))

    fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))
    wedges, texts, autotexts = ax1.pie(
        tipo_cantidad,
        labels=tipo_cantidad.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 10},
        wedgeprops=dict(width=0.5, edgecolor='w', alpha=0.7)
    )
    for text in texts: text.set_fontsize(7)
    for autotext in autotexts: autotext.set_fontsize(6)

    ax1.axis('equal')
    ax1.set_title("Distribución por Tipo de Carbón", fontsize=12)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig1, bbox_inches='tight')

    # === Gráfico de Barras Apiladas por Proveedor ===
    pivot_df = df_sol.pivot_table(index='Proveedor', columns='Tipo', values='Toneladas', aggfunc='sum', fill_value=0)
    pivot_df['Total'] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values('Total', ascending=False)
    valores_df = pivot_df.drop(columns='Total')

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    x = range(len(valores_df))
    bottom = [0] * len(valores_df)
    palette = sns.color_palette("colorblind", n_colors=len(valores_df.columns))

    for i, tipo in enumerate(valores_df.columns):
        valores = valores_df[tipo].values
        ax2.bar(x, valores, bottom=bottom, label=tipo, color=palette[i], alpha=0.7)
        bottom = [bottom[j] + valores[j] for j in range(len(valores))]

    ax2.set_xticks(x)
    ax2.set_xticklabels(valores_df.index, rotation=45, ha='right', fontsize=9)
    ax2.set_title("Pedidos por Proveedor y Tipo de Carbón", fontsize=12)
    ax2.set_ylabel("Toneladas")
    ax2.grid(visible=True, axis='y', linestyle='--', alpha=0.5)
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: format(int(x), ',')))
    ax2.legend(title='Tipo', bbox_to_anchor=(1.01, 1), loc='upper left')

    # Etiquetas de totales
    for i, total in enumerate(pivot_df['Total'].values):
        ax2.text(i, total + total * 0.01, f"{total:,.0f}", ha='center', va='bottom', fontsize=8, rotation=45)

    st.pyplot(fig2)
