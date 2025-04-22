import streamlit as st
import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# 0) Configuração da página
st.set_page_config(
    page_title="Previsão Peso Frangos",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 1) Carregamento de modelos (cache para performance)
@st.cache_resource
def load_models():
    paths = {
        "Ridge A": "ridge_model_A.pkl",
        "Ridge B": "ridge_model_B.pkl",
        "XGB A":   "xgb_model_A.pkl",
        "XGB B":   "xgb_model_B.pkl",
    }
    loaded = {}
    for name, path in paths.items():
        with open(path, "rb") as f:
            loaded[name] = pickle.load(f)
    return loaded

models = load_models()

# 2) Cabeçalho
st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>🪶 Previsão de Peso de Frangos</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

# 3) Cria abas, incluindo página inicial
tab0, tab1, tab2, tab3 = st.tabs([
    "🏠 Início",
    "📈 Modelo A (35, 42, 49)",
    "🌱 Modelo B (42, 49)",
    "🎯 Dia‑Alvo"
])

# Página Inicial com explicações claras
with tab0:
    st.subheader("🏠 Bem-vindo")
    st.markdown(
        "Esta ferramenta auxilia na **previsão de peso de lotes de frangos** e no cálculo do dia em que um peso-alvo será atingido.\n"
        "O usuário deve escolher o fluxo correto de acordo com os dados disponíveis:\n"
        "- **Modelo A**: use quando você **só tiver pesos até o dia 28**. Esse fluxo prevê os dias 35, 42 e 49.\n"
        "- **Modelo B**: use quando você **tiver dados até o dia 35 ou mais**. Esse fluxo prevê apenas os dias 42 e 49, aproveitando informação extra.\n"
        "- **Dia‑Alvo**: estime em qual dia o lote atingirá um peso específico, escolhendo o cenário (A ou B).\n"
        "\n**Para começar**, clique na aba correspondente ao que deseja fazer e siga as instruções na tela."
    )

# Função de plot de curva
def plot_curve(dias, pesos):
    f = interp1d(dias, pesos, fill_value="extrapolate")
    xp = np.arange(min(dias), max(dias) + 1)
    yp = f(xp)
    dfc = pd.DataFrame({"Peso (g)": yp}, index=xp)
    st.line_chart(dfc, use_container_width=True)

# === Aba 1: Modelo A ===
with tab1:
    st.subheader("📊 Previsão para dias 35, 42 e 49")
    st.markdown(
        "**Como usar:** Selecione o algoritmo, preencha os pesos médios até o dia 28 e clique em **Prever A**.\n"
        "Você receberá os pesos previstos para os dias 35, 42 e 49 e verá a curva de crescimento."
    )
    with st.form("form_A"):
        col1, col2 = st.columns(2, gap="medium")
        alg = col1.selectbox("Algoritmo", ["Ridge", "XGB"], key="selA")
        p7  = col1.number_input("Peso dia 7 (g)", min_value=0.0, value=200.0, step=1.0, key="A7")
        p14 = col1.number_input("Peso dia 14 (g)", min_value=0.0, value=500.0, step=1.0, key="A14")
        p21 = col2.number_input("Peso dia 21 (g)", min_value=0.0, value=1000.0, step=1.0, key="A21")
        p28 = col2.number_input("Peso dia 28 (g)", min_value=0.0, value=1500.0, step=1.0, key="A28")
        submitted = st.form_submit_button("▶️ Prever A")
    if submitted:
        with st.spinner("Predizendo..."):
            model = models[f"{alg} A"]
            X_new = np.array([[p7, p14, p21, p28]])
            preds = model.predict(X_new)[0]
            dias  = [35, 42, 49]

        st.markdown("**Resultados**")
        m1, m2, m3 = st.columns(3)
        for col, dia, peso in zip((m1, m2, m3), dias, preds):
            col.metric(label=f"Dia {dia}", value=f"{peso:.1f} g")
        st.markdown("📈 **Curva de Crescimento**")
        plot_curve(dias, preds)

# === Aba 2: Modelo B ===
with tab2:
    st.subheader("📊 Previsão para dias 42 e 49")
    st.markdown(
        "**Como usar:** Selecione o algoritmo, informe os pesos até o dia 35 e a idade do lote, depois clique em **Prever B**.\n"
        "Você receberá os pesos previstos para os dias 42 e 49 e poderá visualizar a curva."
    )
    with st.form("form_B"):
        col1, col2 = st.columns(2, gap="medium")
        alg   = col1.selectbox("Algoritmo", ["Ridge", "XGB"], key="selB")
        p7    = col1.number_input("Peso dia 7 (g)", min_value=0.0, value=200.0, step=1.0, key="B7")
        p14   = col1.number_input("Peso dia 14 (g)", min_value=0.0, value=500.0, step=1.0, key="B14")
        p21   = col2.number_input("Peso dia 21 (g)", min_value=0.0, value=1000.0, step=1.0, key="B21")
        p28   = col2.number_input("Peso dia 28 (g)", min_value=0.0, value=1500.0, step=1.0, key="B28")
        p35   = col1.number_input("Peso dia 35 (g)", min_value=0.0, value=2000.0, step=1.0, key="B35")
        idade = col2.number_input("Idade do lote (dias)", min_value=0, value=45, step=1, key="Bidade")
        submitted = st.form_submit_button("▶️ Prever B")
    if submitted:
        with st.spinner("Predizendo..."):
            model = models[f"{alg} B"]
            X_new = np.array([[p7, p14, p21, p28, p35, idade]])
            preds = model.predict(X_new)[0]
            dias  = [42, 49]

        st.markdown("**Resultados**")
        m1, m2 = st.columns(2)
        for col, dia, peso in zip((m1, m2), dias, preds):
            col.metric(label=f"Dia {dia}", value=f"{peso:.1f} g")
        st.markdown("📈 **Curva de Crescimento**")
        plot_curve(dias, preds)

# === Aba 3: Dia‑Alvo ===
with tab3:
    st.subheader("🎯 Estimativa de Dia‑Alvo para peso desejado")
    st.markdown(
        "**Como usar:** Defina o cenário (A ou B), escolha o algoritmo, preencha os pesos informados até o último dia disponível e o peso‑alvo, então clique em **Calcular Dia‑Alvo**.\n"
        "A aplicação estimará em qual dia o lote atingirá o peso definido e mostrará a curva com o ponto‑alvo."
    )
    with st.form("form_T"):
        scenario = st.radio("Cenário", ["A (até dia 28)", "B (até dia 35)"], horizontal=True, key="sceT")
        alg      = st.selectbox("Algoritmo", ["Ridge", "XGB"], key="selT")
        col1, col2 = st.columns(2, gap="medium")
        p7  = col1.number_input("Peso dia 7 (g)", min_value=0.0, value=200.0, step=1.0, key="T7")
        p14 = col1.number_input("Peso dia 14 (g)", min_value=0.0, value=500.0, step=1.0, key="T14")
        p21 = col2.number_input("Peso dia 21 (g)", min_value=0.0, value=1000.0, step=1.0, key="T21")
        p28 = col2.number_input("Peso dia 28 (g)", min_value=0.0, value=1500.0, step=1.0, key="T28")
        if scenario.startswith("B"):
            p35   = col1.number_input("Peso dia 35 (g)", min_value=0.0, value=2000.0, step=1.0, key="T35")
            idade = col2.number_input("Idade do lote (dias)", min_value=0, value=45, step=1, key="Tidade")
        peso_alvo = st.number_input("Peso‑alvo (g)", min_value=0.0, value=3000.0, step=1.0, key="Talvo")
        submitted = st.form_submit_button("▶️ Calcular Dia‑Alvo")
    if submitted:
        with st.spinner("Calculando..."):
            key   = f"{alg} {'A' if scenario.startswith('A') else 'B'}"
            model = models[key]
            if scenario.startswith("A"):
                X_new = np.array([[p7, p14, p21, p28]])
                dias  = [35, 42, 49]
            else:
                X_new = np.array([[p7, p14, p21, p28, p35, idade]])
                dias  = [42, 49]
            preds  = model.predict(X_new)[0]
            dia_est = float(interp1d(preds, dias, fill_value="extrapolate")(peso_alvo))

        st.markdown("**Dia‑Alvo Estimado**")
        st.write(f"O lote atinge **{peso_alvo:.0f} g** em aproximadamente o dia **{dia_est:.1f}**.")
        st.markdown("📈 **Curva de Crescimento com Ponto‑Alvo**")
        plot_curve(dias, preds)
        st.write(f"**Dia estimado:** {dia_est:.1f} — **Peso‑alvo:** {peso_alvo:.0f} g")
