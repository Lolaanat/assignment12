import streamlit as st
import pickle
import os
import re

# =====================================================
# LOAD MODEL
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "sentiment_model.pkl"
)

with open(MODEL_PATH, "rb") as f:
    model_package = pickle.load(f)

vectorizer = model_package["vectorizer"]
model = model_package["model"]

# =====================================================
# PREPROCESSING
# =====================================================

STOPWORDS_ID = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu',
    'dengan', 'untuk', 'adalah', 'ada', 'atau', 'juga',
    'pada', 'dalam', 'saya', 'kami', 'kita', 'anda',
    'mereka', 'dia', 'ia', 'nya', 'akan', 'sudah',
    'telah', 'tidak', 'bukan', 'jangan', 'tak',
    'lebih', 'sangat', 'sekali', 'paling', 'bisa',
    'dapat', 'seperti', 'saat', 'serta', 'oleh',
    'agar', 'tetapi', 'namun', 'karena', 'jika',
    'kalau', 'maka', 'lalu', 'kemudian', 'pun',
    'lah', 'kah', 'deh', 'dong', 'sih', 'kok',
    'ya', 'yah', 'oh', 'eh', 'ah', 'ih', 'uh',
    'wah', 'hah', 'hem', 'hmm'
}


def preprocess_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r'http\S+|www\S+', '', text)

    text = re.sub(r'@\w+|#\w+', '', text)

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()

    tokens = [
        word
        for word in tokens
        if word not in STOPWORDS_ID
        and len(word) > 2
    ]

    return " ".join(tokens)

# =====================================================
# UI
# =====================================================

st.set_page_config(
    page_title="Analisis Sentimen",
    page_icon="😊",
    layout="centered"
)

st.title("😊 Analisis Sentimen Review")

st.write(
    "Masukkan kalimat atau review untuk mengetahui sentimennya."
)

text = st.text_area(
    "Masukkan Kalimat / Review",
    height=150
)

# =====================================================
# PREDIKSI
# =====================================================

if st.button("Prediksi"):

    if text.strip() == "":
        st.warning("Silakan masukkan kalimat terlebih dahulu.")
    else:

        clean_text = preprocess_text(text)

        text_vec = vectorizer.transform([clean_text])

        pred = model.predict(text_vec)[0]

        # ===============================
        # SESUAIKAN LABEL DATASETMU
        # ===============================
        # Jika:
        # 0 = Positif
        # 1 = Negatif

        if pred == 0:
            st.success("😌 Sentimen Positif")
        else:
            st.error("😔 Sentimen Negatif")
