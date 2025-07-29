import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import joblib

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi CNN+LSTM", layout="wide")
st.title("📈 Prediksi Time Series dengan CNN + LSTM")

# Load model & scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model("cnn_lstm_model.h5")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_model_and_scaler()

# Fungsi membuat sequence sliding window
def create_sequences(data, window_size=60):
    X = []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
    return np.array(X)

# Upload file CSV
uploaded_file = st.file_uploader("Unggah file CSV dengan kolom 'ddate' dan 'tag_value'", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'ddate' not in df.columns or 'tag_value' not in df.columns:
        st.error("CSV harus memiliki kolom 'ddate' dan 'tag_value'.")
    else:
        # Ubah timestamp
        df['ddate'] = pd.to_datetime(df['ddate'])

        # Sort data jika belum urut
        df = df.sort_values('ddate')

        # Normalisasi
        scaled_data = scaler.transform(df[['tag_value']])

        # Buat window input
        window_size = 60
        X_input = create_sequences(scaled_data, window_size)

        if len(X_input) == 0:
            st.warning("Data terlalu sedikit untuk dibuat sequence (minimal 60 baris).")
        else:
            # Prediksi
            y_pred_scaled = model.predict(X_input)
            y_pred = scaler.inverse_transform(y_pred_scaled)

            # Gabungkan ke DataFrame (disesuaikan offset window)
            df_pred = df.iloc[window_size:].copy()
            df_pred['predicted'] = y_pred

            # Tampilkan data tabel
            st.subheader("🗃️ Data Asli dan Prediksi")
            st.dataframe(df_pred[['ddate', 'tag_value', 'predicted']].head(20))

            # Plot
            st.subheader("📊 Visualisasi Prediksi vs Data Aktual")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_pred['ddate'], df_pred['tag_value'], label="Aktual", color='blue')
            ax.plot(df_pred['ddate'], df_pred['predicted'], label="Prediksi", color='orange')
            ax.set_xlabel("Waktu")
            ax.set_ylabel("Nilai")
            ax.legend()
            st.pyplot(fig)
else:
    st.info("Silakan unggah file CSV untuk melakukan prediksi.")
