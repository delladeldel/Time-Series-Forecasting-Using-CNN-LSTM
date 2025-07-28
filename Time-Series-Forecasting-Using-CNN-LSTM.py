import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Set page
st.set_page_config(page_title="Time Series Forecasting", layout="wide")

# Load model & scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model("cnn_lstm_model.keras")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_model_and_scaler()

st.title("📈 Prediksi Tag Value 10 Menit ke Depan (CNN-LSTM)")

st.markdown("""
Silakan masukkan data `tag_value` terakhir sebanyak **60 data poin** (karena model memerlukan window 60), 
dengan interval 10 detik antar data poin.
""")

# Input manual atau upload
option = st.radio("Pilih metode input:", ["Manual", "Upload CSV"], horizontal=True)

if option == "Manual":
    data_input = st.text_area("Masukkan 60 nilai tag_value (pisahkan dengan koma):", height=150)
    try:
        data = [float(i.strip()) for i in data_input.split(",") if i.strip()]
        if len(data) != 60:
            st.error("Jumlah data harus tepat 60.")
            st.stop()
    except ValueError:
        st.error("Pastikan semua nilai berupa angka.")
        st.stop()

elif option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload file CSV (kolom: tag_value)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "tag_value" not in df.columns:
            st.error("CSV harus memiliki kolom bernama 'tag_value'")
            st.stop()
        if len(df) < 60:
            st.error("Minimal 60 data diperlukan.")
            st.stop()
        data = df["tag_value"].values[-60:]

# Prediksi
if st.button("🔮 Prediksi 10 Menit ke Depan"):
    # Scaling dan reshape
    scaled_data = scaler.transform(np.array(data).reshape(-1, 1)).flatten()
    input_data = scaled_data.reshape((1, 60, 1))

    # Prediksi
    prediction = model.predict(input_data)[0]
    predicted_values = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()

    # Tampilkan hasil
    st.subheader("📊 Hasil Prediksi")

    fig, ax = plt.subplots(figsize=(10, 4))
    time = np.arange(1, len(predicted_values) + 1) * 10  # 10 detik interval
    ax.plot(time, predicted_values, marker='o', label="Prediksi", color='orange')
    ax.set_xlabel("Detik ke-")
    ax.set_ylabel("Tag Value")
    ax.set_title("Prediksi 10 Menit (60 titik data, tiap 10 detik)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Dataframe hasil
    result_df = pd.DataFrame({
        "Waktu (detik ke-)": time,
        "Prediksi Tag Value": predicted_values
    })
    st.dataframe(result_df)
