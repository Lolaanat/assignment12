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
    'yang','dan','di','ke','dari','ini','itu',
    'dengan','untuk','adalah','ada','atau','juga',
    'pada','dalam','saya','kami','kita','anda',
    'mereka','dia','ia','nya','akan','sudah',
    'telah'
}

NEGATIVE_WORDS = {
    "tidak",
    "tak",
    "bukan",
    "jangan"
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
# CEK NEGASI
# =====================================================

def contains_negative_word(text):

    text = text.lower()

    words = text.split()

    for word in words:
        if word in NEGATIVE_WORDS:
            return True

    return False


# =====================================================
# UI
# =====================================================

st.set_page_config(
    page_title="Analisis Sentimen",
    page_icon="😊",
    layout="centered"
)

st.title("Analisis Sentimen Review 🌟")

st.caption("Ini merupakan web analisis sentimen Leora Natania - 2702217195.")

text = st.text_area(
    "Masukkan kalimat atau review untuk mengetahui sentimennya.",
    height=150
)

# =====================================================
# PREDIKSI
# =====================================================

if st.button("Prediksi"):

    if text.strip() == "":
        st.warning("Silakan masukkan kalimat terlebih dahulu.")

    else:

        # RULE PRIORITAS
        # Jika ada kata negasi -> langsung negatif

        if contains_negative_word(text):

            st.error("😔 Sentimen Negatif")

        else:

            clean_text = preprocess_text(text)

            text_vec = vectorizer.transform([clean_text])

            pred = model.predict(text_vec)[0]

            # 0 = Positif
            # 1 = Negatif

            if pred == 0:
                st.error("😔 Sentimen Negatif")
            else:
                st.success("😌 Sentimen Positif")
