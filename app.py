import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd

# 0) Configuração da página
st.set_page_config(
    page_title="Previsão Peso Frangos - ETR",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1) Carregamento de modelos ETR
@st.cache_resource
def load_models():
    modelos = {}
    pasta = 'modelos_etr/'
    for arquivo in os.listdir(pasta):
        if arquivo.endswith('.pkl'):
            dia = arquivo.replace('ETR_model_', '').replace('.pkl', '')
            with open(os.path.join(pasta, arquivo), 'rb') as f:
                modelos[dia] = pickle.load(f)
    return modelos

models = load_models()

# 2) Cabeçalho
st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>Previsão de Peso de Frangos (ETR)</h1>
""", unsafe_allow_html=True)
st.markdown("---")

# 3) Formulário para preenchimento
st.subheader("Informe os Pesos Até o Dia 35")
with st.form("form_pesos"):
    col1, col2, col3 = st.columns(3)
    p7 = col1.number_input("Peso Dia 7 (g)", min_value=0.0, value=200.0, step=1.0)
    p14 = col1.number_input("Peso Dia 14 (g)", min_value=0.0, value=500.0, step=1.0)
    p21 = col2.number_input("Peso Dia 21 (g)", min_value=0.0, value=1000.0, step=1.0)
    p28 = col2.number_input("Peso Dia 28 (g)", min_value=0.0, value=1500.0, step=1.0)
    p35 = col3.number_input("Peso Dia 35 (g)", min_value=0.0, value=2000.0, step=1.0)
    submit = st.form_submit_button("▶️ Prever Pesos")

# 4) Previsão
if submit:
    with st.spinner("Calculando previsões..."):
        X_new = np.array([[p7, p14, p21, p28, p35]])
        previsoes = {}
        for dia, model in models.items():
            pred = model.predict(X_new)[0]
            previsoes[dia] = pred

    # Mostrar resultados ordenados por dia
    st.subheader("📊 Resultados de Previsão")
    previsoes_ordenadas = dict(sorted(previsoes.items(), key=lambda x: int(x[0].split(' ')[-1].replace('+', ''))))
    df_preds = pd.DataFrame({"Dia": list(previsoes_ordenadas.keys()), "Peso Previsto (g)": list(previsoes_ordenadas.values())})
    st.dataframe(df_preds, use_container_width=True)

    # Gráfico
    st.markdown("""### 📊 Curva de Crescimento Prevista""")
    st.line_chart(df_preds.set_index('Dia'))
