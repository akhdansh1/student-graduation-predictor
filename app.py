import streamlit as st
import pandas as pd
import joblib

# Load model dan encoders
model = joblib.load("model_rf.pkl")
encoders = joblib.load("label_encoder.pkl")

st.title("Prediksi Kelulusan Mahasiswa")

# Ambil nama kolom input dari encoder yang sudah disimpan
input_columns = list(encoders.keys())

# Buat form input
user_input = {}
for col in input_columns:
    options = encoders[col].classes_
    user_input[col] = st.selectbox(f"Pilih nilai untuk '{col}':", options)

# Fitur numerik tambahan (dari dataset asli)
numerical_fields = [
    "raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"
]

for col in numerical_fields:
    user_input[col] = st.number_input(f"Masukkan nilai untuk '{col}':", min_value=0)

# Prediksi
if st.button("Prediksi"):
    try:
        # Bentuk dataframe
        input_df = pd.DataFrame([user_input])

        # Encode kolom kategorikal
        for col in input_columns:
            val = input_df[col].iloc[0]
            if val in encoders[col].classes_:
                input_df[col] = encoders[col].transform([val])
            else:
                st.warning(f"⚠️ Nilai '{val}' di kolom '{col}' tidak dikenal oleh encoder. Gunakan nilai yang tersedia.")
                st.stop()

        # Pastikan semua kolom sesuai urutan fit model
        final_input = input_df[model.feature_names_in_]

        # Prediksi
        pred = model.predict(final_input)[0]
        prob = model.predict_proba(final_input)[0][pred]

        label = "Lulus Tepat Waktu" if pred == 1 else "Tidak Lulus Tepat Waktu"
        st.success(f"Hasil Prediksi: **{label}** ({prob*100:.2f}% kemungkinan)")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses input: {e}")
