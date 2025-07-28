import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Set halaman Streamlit
st.set_page_config(page_title="Time Series Forecasting", layout="wide")

# Load model & scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model("cnn_lstm_model.h5")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_model_and_scaler()

st.title("📈 Prediksi Tag Value 10 Menit ke Depan (CNN-LSTM)")
st.markdown("""
Silakan masukkan data `tag_value` terakhir sebanyak **60 data poin** 
(dengan interval 10 detik antar data poin).
""")

# Inisialisasi
data = None

# Input: Manual atau Upload
option = st.radio("Pilih metode input:", ["Manual", "Upload CSV"], horizontal=True)

if option == "Manual":
    data_input = st.text_area("Masukkan 60 nilai tag_value (pisahkan dengan koma):", height=150)
    if data_input:
        try:
            data = [float(i.strip()) for i in data_input.split(",") if i.strip()]
            if len(data) != 60:
                st.error("Jumlah data harus tepat 60.")
                data = None
        except ValueError:
            st.error("Pastikan semua nilai berupa angka.")
            data = None

elif option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload file CSV (harus punya kolom: tag_value)", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "tag_value" not in df.columns:
                st.error("Kolom 'tag_value' tidak ditemukan.")
            elif len(df) < 60:
                st.error("CSV harus memiliki minimal 60 data.")
            else:
                data = df["tag_value"].values[-60:]
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            data = None

# Prediksi jika data valid
if st.button("🔮 Prediksi 10 Menit ke Depan"):
    if data is None:
        st.warning("Silakan masukkan data yang valid terlebih dahulu.")
        st.stop()

    try:
        # Preprocessing
        scaled_data = scaler.transform(np.array(data).reshape(-1, 1)).flatten()
        input_seq = list(scaled_data)  # list agar bisa diappend terus-menerus

        predicted_scaled = []

        # Lakukan prediksi autoregressive sebanyak 60 kali
        for i in range(60):
            last_60 = np.array(input_seq[-60:]).reshape(1, 60, 1)
            pred = model.predict(last_60, verbose=0)[0][0]
            predicted_scaled.append(pred)
            input_seq.append(pred)

        # Invers transform ke nilai asli
        predicted_values = scaler.inverse_transform(np.array(predicted_scaled).reshape(-1, 1)).flatten()

        # Visualisasi
        st.subheader("📊 Hasil Prediksi")
        fig, ax = plt.subplots(figsize=(10, 4))
        time = np.arange(1, 61) * 10  # 10 detik interval untuk 60 titik data
        ax.plot(time, predicted_values, marker='o', label="Prediksi", color='orange')
        ax.set_xlabel("Detik ke-")
        ax.set_ylabel("Tag Value")
        ax.set_title("Prediksi 10 Menit (60 titik data, tiap 10 detik)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # Tampilkan hasil dalam tabel
        result_df = pd.DataFrame({
            "Waktu (detik ke-)": time,
            "Prediksi Tag Value": predicted_values
        })
        st.dataframe(result_df)

        # Tambahkan tombol download
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Hasil Prediksi sebagai CSV",
            data=csv,
            file_name='hasil_prediksi.csv',
            mime='text/csv',
        )

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
