import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model

st.set_page_config(page_title="Forecast CNN+LSTM", layout="wide")
st.title("📈 Forecast 60 Langkah ke Depan dengan CNN + LSTM")

# ====== Load model & scaler dari lokal ======
MODEL_PATH = "cnn_lstm_model.keras"
SCALER_PATH = "scaler.joblib"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ====== Upload CSV dari pengguna ======
csv_file = st.file_uploader("📤 Upload Data CSV (berisi 'ddate' & 'tag_value')", type=["csv"])

if csv_file:
    df = pd.read_csv(csv_file)
    df['ddate'] = pd.to_datetime(df['ddate'])
    df = df.sort_values('ddate')
    st.success("✅ Data berhasil dimuat.")
    st.dataframe(df.tail(10))

    # Pilih window size
    window_size = st.slider("🎚️ Pilih Window Size:", min_value=10, max_value=200, value=60, step=10)

    # Normalisasi data dan ambil window terakhir
    scaled_data = scaler.transform(df[['tag_value']])
    last_window = scaled_data[-window_size:].reshape(1, window_size, 1)

    # Forecast 60 langkah ke depan
    steps_ahead = 60
    preds = []

    for _ in range(steps_ahead):
        pred_scaled = model.predict(last_window, verbose=0)
        pred_inv = scaler.inverse_transform(pred_scaled)[0][0]
        preds.append(pred_inv)
        last_window = np.append(last_window[:, 1:, :], [[[pred_scaled[0][0]]]], axis=1)

    # Buat timestamp prediksi ke depan
    time_interval = df['ddate'].diff().dropna().mode()[0]
    last_time = df['ddate'].iloc[-1]
    future_dates = [last_time + (i + 1) * time_interval for i in range(steps_ahead)]

    # Buat dataframe hasil
    df_future = pd.DataFrame({
        'ddate': future_dates,
        'predicted': preds
    })

    # ====== Tampilkan hasil ======
    st.subheader("🗃️ Hasil Forecast 60 Langkah ke Depan")
    st.dataframe(df_future)

    # ====== Grafik ======
    st.subheader("📊 Grafik Prediksi Masa Depan")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['ddate'], df['tag_value'], label="Data Historis", color='blue')
    ax.plot(df_future['ddate'], df_future['predicted'], label="Forecast", color='red')
    ax.set_xlabel("Waktu")
    ax.set_ylabel("Nilai")
    ax.legend()
    st.pyplot(fig)

    # ====== Download ======
    st.subheader("📥 Unduh Hasil Forecast")
    forecast_csv = df_future.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download sebagai CSV", data=forecast_csv, file_name="forecast_60_langkah.csv", mime="text/csv")
else:
    st.info("Silakan upload file CSV terlebih dahulu.")
