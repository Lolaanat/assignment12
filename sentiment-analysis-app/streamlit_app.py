import streamlit as st
import pickle
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "sentiment_model.pkl"
)

with open(MODEL_PATH, "rb") as f:
    model_package = pickle.load(f)

vectorizer = model_package["vectorizer"]
model = model_package["model"]

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
    "Masukkan Review",
    height=150
)

if st.button("Prediksi Sentimen"):

    if text.strip() == "":
        st.warning("Masukkan teks terlebih dahulu.")
    else:

        # preprocessing
        clean_text = preprocess_text(text)

        # debug
        st.write("### Hasil Preprocessing")
        st.code(clean_text)

        # cek apakah ada kata yang dikenali model
        known_words = [
            word
            for word in clean_text.split()
            if word in vectorizer.vocabulary_
        ]

        st.write("### Kata yang dikenali model")
        st.write(known_words)

        if len(known_words) == 0:
            st.error(
                "Model tidak mengenali kata-kata pada input ini. "
                "Coba gunakan kalimat yang mirip dengan data training."
            )

        else:

            text_vec = vectorizer.transform([clean_text])

            pred = model.predict(text_vec)[0]

            st.write("### Hasil Prediksi")

            if pred == 1:
                st.success("😊 Sentimen Positif")
            else:
                st.error("😞 Sentimen Negatif")
