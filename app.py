import streamlit as st
import numpy as np
import joblib

modelo = joblib.load('modelo_calorias_rf.pkl')

st.title('Previsão de Calorias Queimadas')

# Entradas
fat_percentage = st.number_input('Percentual de Gordura Corporal (%)', min_value=0.0, max_value=100.0, value=20.0)
weight = st.number_input('Peso (kg)', min_value=0.0, value=70.0)

# Entrada de duração: horas e minutos
st.subheader('Duração da Sessão')
horas = st.number_input('Horas', min_value=0, max_value=5, value=1)
minutos = st.number_input('Minutos', min_value=0, max_value=59, value=0)
session_duration = horas + (minutos / 60)

height = st.number_input('Altura (m)', min_value=0.0, value=1.70)
avg_bpm = st.number_input('Média de BPM durante o treino', min_value=0.0, value=120.0)
age = st.number_input('Idade', min_value=0, value=25)
water_intake = st.number_input('Consumo de Água (litros)', min_value=0.0, value=2.0)
resting_bpm = st.number_input('BPM em repouso', min_value=0.0, value=70.0)

# Previsão
if st.button('Prever Calorias Queimadas'):
    entrada = np.array([[fat_percentage, weight, session_duration, height,
                         avg_bpm, age, water_intake, resting_bpm]])
    predicao = modelo.predict(entrada)
    st.success(f'Calorias estimadas por sessão: {predicao[0]:.2f}')
