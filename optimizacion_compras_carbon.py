import streamlit as st
import pandas as pd
from collections import OrderedDict
from pulp import *
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

sns.set_theme(style="whitegrid")  # Otros estilos: "dark", "ticks", "white"

st.set_page_config(page_title="Modelo de Optimización de Carbón", layout="wide")

st.title("🧮 Modelo de Optimización de Compras de Carbón")
st.markdown("Minimiza el costo total de compra de carbón cumpliendo restricciones de calidad, mezcla y disponibilidad.")

# 📁 Cargar archivo
archivo = st.file_uploader("📤 Carga el archivo de datos (.xlsx):", type=["xlsx"])
if archivo:
    hoja = pd.read_excel(archivo, sheet_name=0, header=None)

    # --- Procesamiento de datos ---
    df = hoja.iloc[2:, 0:9].copy()
    df.columns = hoja.iloc[1, 0:9]

    for col in ['Disponible', 'Precio', 'HT', 'CZ', 'MV', 'S', 'FSI']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[df['Disponible'] > 0]
    df = df[df['Precio'].notnull()]
    df = df.sort_values(['Proveedor', 'Tipo'])
    df = df.drop_duplicates(subset=['Proveedor', 'Tipo'], keep='first')

    necesidad = hoja.iloc[2, 10]
    calidad_esperada = hoja.iloc[2, 12:16]
    calidad_esperada.index = hoja.iloc[1, 12:16]
    requerimiento_total = float(necesidad)

    limites = hoja.iloc[2:, 17:19]
    limites.columns = hoja.iloc[1, 17:19]
    limites = limites.dropna()

    proveedores = df['Proveedor'].unique().tolist()
    tipos = df['Tipo'].unique().tolist()
    pares = list(OrderedDict.fromkeys(zip(df['Proveedor'], df['Tipo'])))

    precio = {(prov, tipo): row['Precio'] for prov, tipo, row in zip(df['Proveedor'], df['Tipo'], df.to_dict('records'))}
    disponible = {(prov, tipo): row['Disponible'] for prov, tipo, row in zip(df['Proveedor'], df['Tipo'], df.to_dict('records'))}
    ht = {(prov, tipo): row['HT'] for prov, tipo, row in zip(df['Proveedor'], df['Tipo'], df.to_dict('records'))}
    cz = {(prov, tipo): row['CZ'] for prov, tipo, row in zip(df['Proveedor'], df['Tipo'], df.to_dict('records'))}
    mv = {(prov, tipo): row['MV'] for prov, tipo, row in zip(df['Proveedor'], df['Tipo'], df.to_dict('records'))}
    s = {(prov, tipo): row['S'] for prov, tipo, row in zip(df['Proveedor'], df['Tipo'], df.to_dict('records'))}
    fsi = {(prov, tipo): row['FSI'] for prov, tipo, row in zip(df['Proveedor'], df['Tipo'], df.to_dict('records'))}
    calidad_esperada_dict = calidad_esperada.to_dict()
    limites_dict = dict(zip(limites['TIPO'], limites['LIMITE']))

    # === Modelo ===
    model = LpProblem("Optimizacion_Compras_Carbon", LpMinimize)
    x = LpVariable.dicts("Pedido", pares, lowBound=0, cat='Continuous')
    model += lpSum(x[par] * precio[par] for par in pares)

    # Restricciones
    model += lpSum(x[par] for par in pares) == requerimiento_total, "Requerimiento_Total"
    for par in pares:
        model += x[par] <= disponible[par], f"Disponible_{par}"

    for tipo in tipos:
        limite = limites_dict.get(tipo, 1)
        model += lpSum(x[par] for par in pares if par[1] == tipo) <= limite * requerimiento_total, f"PorcentajeMax_{tipo}"

    model += lpSum(x[par] * s[par] for par in pares) <= calidad_esperada_dict['S'] * lpSum(x[par] for par in pares), "S_esperado"
    model += lpSum(x[par] * fsi[par] for par in pares) >= calidad_esperada_dict['FSI'] * lpSum(x[par] for par in pares), "FSI_esperado"
    model += lpSum(x[par] * cz[par] for par in pares) <= calidad_esperada_dict['CZ'] * lpSum(x[par] for par in pares), "CZ_esperado"
    model += lpSum(x[par] * mv[par] for par in pares) <= calidad_esperada_dict['MV'] * lpSum(x[par] for par in pares), "MV_esperado"

    # === Resolver ===
    with st.spinner("🔄 Ejecutando modelo..."):
        model.solve()

    st.success("✅ Modelo resuelto")
    st.write(f"**Estado:** {LpStatus[model.status]}")
    st.write(f"**Costo Total:** ${value(model.objective):,.0f}")

    # === Mostrar Resultados ===
    solucion = {par: x[par].varValue for par in pares if x[par].varValue > 0}
    df_sol = pd.DataFrame([
        {"Proveedor": prov, "Tipo": tipo, "Toneladas": cantidad}
        for (prov, tipo), cantidad in solucion.items()
    ])
    st.dataframe(df_sol, use_container_width=True)

    # # === Gráfico de Torta por Tipo ===
    # tipo_cantidad = df_sol.groupby("Tipo")["Toneladas"].sum()
    # fig1, ax1 = plt.subplots(figsize=(4, 4))  # reducido
    # ax1.pie(tipo_cantidad, labels=tipo_cantidad.index, autopct='%1.1f%%', startangle=90)
    # ax1.axis('equal')
    # ax1.set_title("Distribución por Tipo de Carbón", fontsize=14)
    
    # col1, col2, col3 = st.columns([1, 2, 1])  # La columna del medio es más ancha
    # with col2:
    #     st.pyplot(fig1, bbox_inches='tight')

    # # === Gráfico de Barras Apiladas ===
    # pivot_df = df_sol.pivot_table(index='Proveedor', columns='Tipo', values='Toneladas', aggfunc='sum', fill_value=0)
    # pivot_df['Total'] = pivot_df.sum(axis=1)
    # pivot_df = pivot_df.sort_values('Total', ascending=False)
    # pivot_df_values = pivot_df.drop(columns='Total')

    # fig2, ax2 = plt.subplots(figsize=(12, 6))
    # pivot_df_values.plot(kind='bar', stacked=True, ax=ax2, colormap='tab20')
    # ax2.set_title("Pedidos por Proveedor y Tipo de Carbón", fontsize=14)
    # ax2.set_ylabel("Toneladas")
    # ax2.grid(axis='y', linestyle='--', alpha=0.7)
    # ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',')))

    # # etiquetas en diagonal
    # plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')


    # for i, total in enumerate(pivot_df['Total'].values):
    #     ax2.text(i, total + total*0.01, f"{total:,.0f}", ha='center', va='bottom', fontsize=9, rotation=45)
    # st.pyplot(fig2)
    
    # === Gráfico de Torta por Tipo ===
    tipo_cantidad = df_sol.groupby("Tipo")["Toneladas"].sum()
    colors = sns.color_palette("pastel")[0:len(tipo_cantidad)]  # Colores suaves

    fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))
    wedges, texts, autotexts = ax1.pie(
        tipo_cantidad,
        labels=tipo_cantidad.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 10},
        wedgeprops=dict(width=0.5, edgecolor='w')  # estilo donut
    )
    ax1.axis('equal')
    ax1.set_title("Distribución por Tipo de Carbón", fontsize=14)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig1, bbox_inches='tight')
        
    # === Gráfico de Barras Apiladas ===
    # === Preparación del DataFrame ===
    pivot_df = df_sol.pivot_table(index='Proveedor', columns='Tipo', values='Toneladas', aggfunc='sum', fill_value=0)
    pivot_df['Total'] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values('Total', ascending=False)
    pivot_df_values = pivot_df.drop(columns='Total')

    # Paleta de colores más agradable
    tipo_list = pivot_df_values.columns
    palette = sns.color_palette("pastel", n_colors=len(tipo_list))  # colores más suaves y armoniosos

    fig2, ax2 = plt.subplots(figsize=(12, 6))

    bottom = [0] * len(pivot_df_values)
    x = range(len(pivot_df_values))

    for i, tipo in enumerate(tipo_list):
        valores = pivot_df_values[tipo].values
        ax2.bar(x, valores, bottom=bottom, label=tipo, color=palette[i])
        bottom = [bottom[j] + valores[j] for j in range(len(valores))]

    ax2.set_xticks(x)
    ax2.set_xticklabels(pivot_df_values.index, rotation=45, ha='right')
    ax2.set_title("Pedidos por Proveedor y Tipo de Carbón", fontsize=14)
    ax2.set_ylabel("Toneladas")
    ax2.set_xlabel("")

    # Solo líneas horizontales suaves
    ax2.grid(visible=True, axis='x', linestyle='--', alpha=0.5)
    # ax2.grid(visible=True, axis='y', linestyle='--', alpha=0.5)

    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax2.legend(title='Tipo', bbox_to_anchor=(1.01, 1), loc='upper left')

    # Etiquetas de totales encima de cada barra
    totales = pivot_df['Total'].values
    for i, total in enumerate(totales):
        ax2.text(i, total + total * 0.01, f"{total:,.0f}", ha='center', va='bottom', fontsize=9, rotation=45)

    st.pyplot(fig2)
