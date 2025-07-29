import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import joblib
import os
from keras.models import model_from_json

st.set_page_config(page_title="Prediksi CNN+LSTM", layout="wide")
st.title("📈 Prediksi Time Series dengan CNN + LSTM")

# Upload semua file yang dibutuhkan
json_file = st.file_uploader("Upload Struktur Model (model_structure.json)", type=["json"])
weights_file = st.file_uploader("Upload Bobot Model (model_weights.h5)", type=["h5"])
scaler_file = st.file_uploader("Upload Scaler (.joblib)", type=["joblib"])
csv_file = st.file_uploader("Upload Data CSV (berisi 'ddate' dan 'tag_value')", type=["csv"])

# Cek semua file sudah diunggah
if json_file and weights_file and scaler_file and csv_file:
    # Simpan file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_json:
        tmp_json.write(json_file.read())
        json_path = tmp_json.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_weights:
        tmp_weights.write(weights_file.read())
        weights_path = tmp_weights.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp_scaler:
        tmp_scaler.write(scaler_file.read())
        scaler_path = tmp_scaler.name

    # Load model structure
    with open(json_path, "r") as f:
        model_json = f.read()
    model = model_from_json(model_json)

    # Load model weights
    model.load_weights(weights_path)

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Load data
    df = pd.read_csv(csv_file)
    df['ddate'] = pd.to_datetime(df['ddate'])
    df = df.sort_values('ddate')

    # Normalize data
    scaled_data = scaler.transform(df[['tag_value']])

    # Buat window input
    def create_sequences(data, window_size=60):
        return np.array([data[i-window_size:i] for i in range(window_size, len(data))])

    window_size = 60
    X_input = create_sequences(scaled_data, window_size)

    if len(X_input) == 0:
        st.warning("Jumlah data kurang dari window size (60). Tidak dapat diprediksi.")
    else:
        # Prediksi
        y_pred_scaled = model.predict(X_input)
        y_pred = scaler.inverse_transform(y_pred_scaled)

        # Gabungkan prediksi ke dataframe
        df_pred = df.iloc[window_size:].copy()
        df_pred['predicted'] = y_pred

        # Tampilkan data tabel
        st.subheader("🗃️ Data dan Hasil Prediksi")
        st.dataframe(df_pred[['ddate', 'tag_value', 'predicted']].head(20))

        # Visualisasi
        st.subheader("📊 Grafik Perbandingan")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_pred['ddate'], df_pred['tag_value'], label="Aktual", color='blue')
        ax.plot(df_pred['ddate'], df_pred['predicted'], label="Prediksi", color='orange')
        ax.set_xlabel("Waktu")
        ax.set_ylabel("Nilai")
        ax.legend()
        st.pyplot(fig)

    # Bersihkan file temp
    os.remove(json_path)
    os.remove(weights_path)
    os.remove(scaler_path)

else:
    st.info("Silakan upload keempat file: struktur model (.json), bobot model (.h5), scaler (.joblib), dan data (.csv)")
