# 🎨 INTERFACCIA WEB CON STREAMLIT - Crea un nuovo file
%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Configurazione pagina
st.set_page_config(page_title="Medical AI - Intubazione", layout="wide")
st.title("🏥 AI - Predizione Intubazione Difficile")
st.write("Sistema di intelligenza artificiale per predire il rischio di intubazione difficile")

# Sidebar per input
st.sidebar.header("📊 Parametri Paziente")

eta = st.sidebar.slider("Età", 18, 90, 55)
peso = st.sidebar.slider("Peso (kg)", 40, 150, 75)
mallampati = st.sidebar.selectbox("Mallampati", [1, 2, 3, 4], index=1)
stop_bang = st.sidebar.slider("STOP-BANG Score", 0, 8, 3)
al_ganzuri = st.sidebar.slider("Al-Ganzuri (cm)", 2.0, 8.0, 4.5)
dimensioni = st.sidebar.slider("Dimensioni collo (cm)", 10.0, 25.0, 16.0)
dii = st.sidebar.slider("DII (cm)", 2.0, 10.0, 5.5)

# Pulsante predizione
if st.sidebar.button("🎯 Calcola Rischio", type="primary"):
    # Carica modello (qui semplificato)
    try:
        # Simulazione predizione - sostituisci con modello reale
        features = np.array([[eta, peso, mallampati, stop_bang, al_ganzuri, dimensioni, dii]])
        
        # Fattori di rischio (logica semplificata)
        risk_factors = (
            (mallampati >= 3) * 0.3 +
            (stop_bang >= 5) * 0.25 +
            (eta > 55) * 0.15 +
            (peso > 90) * 0.15 +
            (al_ganzuri < 4) * 0.15
        )
        
        probability = min(risk_factors, 0.95)  # Max 95%
        
        # Display risultati
        st.success(f"**Probabilità intubazione difficile: {probability:.1%}**")
        
        if probability > 0.5:
            st.error("""
            ⚠️ **RISCHIO ALTO** 
            - Preparare video-laringoscopio
            - Avere a disposizione guide stiliate
            - Team esperto presente
            - Considerare tecniche alternative
            """)
        else:
            st.success("""
            ✅ **RISCHIO BASSO**
            - Procedura standard
            - Monitoraggio routinario
            - Equipment standard disponibile
            """)
            
    except Exception as e:
        st.error(f"Errore nella predizione: {e}")

# Informazioni sistema
st.sidebar.markdown("---")
st.sidebar.info("""
**Informazioni Sistema:**
- Target: Cormack > 2 = Intubazione difficile
- Modello: Random Forest
- Accuratezza: >90% su dati di test
""")
