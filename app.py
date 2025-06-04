import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Load data menu
df = pd.read_csv('content/menu_kopi.csv')
menu = df.to_dict(orient='records')

# Corpus untuk pencarian
corpus = [
    f"{item['nama']} {item['kategori']} {item['deskripsi']}" for item in menu
]

# Vectorizer untuk pencarian menu
vectorizer_corpus = TfidfVectorizer(
    analyzer='word', ngram_range=(1, 2), lowercase=True)
tfidf_matrix_corpus = vectorizer_corpus.fit_transform(corpus)

# Korpus kata unik dari nama, kategori, deskripsi
unique_words = set()
for item in menu:
    unique_words.update(item['nama'].lower().split())
    unique_words.update(item['kategori'].lower().split())
    unique_words.update(item['deskripsi'].lower().split())
unique_vocab = list(unique_words)

# Vectorizer khusus untuk saran kata (per kata, bukan dokumen)
# cocokkan typo berdasarkan bentuk
vectorizer_words = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
word_vectors = vectorizer_words.fit_transform(unique_vocab)

# Log pencarian
search_log = []

# Fungsi pencarian menu


def search_menu(query, kategori_filter=None):
    query_vec = vectorizer_corpus.transform([query.lower()])
    scores = cosine_similarity(query_vec, tfidf_matrix_corpus).flatten()

    results = []
    for idx, score in enumerate(scores):
        item = menu[idx]
        if kategori_filter and kategori_filter.lower() != 'semua' and item['kategori'].lower() != kategori_filter.lower():
            continue
        if score > 0:
            results.append((item, score))

    results.sort(key=lambda x: -x[1])
    return results

# Fungsi saran kata berdasarkan cosine similarity


def suggest_closest_words(query, top_n=3):
    query_vec = vectorizer_words.transform([query.lower()])
    similarities = cosine_similarity(query_vec, word_vectors).flatten()
    similar_words = sorted(
        zip(unique_vocab, similarities), key=lambda x: -x[1])
    return [word for word, score in similar_words if score > 0][:top_n]

# Fungsi pencarian populer


def show_popular_queries(log, top_n=5):
    return Counter(log).most_common(top_n)


# Streamlit UI
st.title("📋 Pencari Menu Kopi")
st.markdown("Masukkan kata kunci (dalam Bahasa Indonesia) untuk mencari menu kopi berdasarkan nama, kategori, atau deskripsi.")

query = st.text_input("🔍 Kata kunci")
kategori = st.selectbox("☕ Kategori", options=['Semua', 'Hot', 'Cold'])

col1, col2 = st.columns(2)

if col1.button("Cari"):
    if query.strip():
        search_log.append(query.strip())
        hasil = search_menu(query, kategori_filter=kategori)

        if hasil:
            st.subheader(
                f"Hasil pencarian untuk: '{query}' (Kategori: {kategori})")
            for item, score in hasil:
                st.markdown(
                    f"**{item['nama']}** ({item['kategori']})  \n{item['deskripsi']}  \n_Skor: {score:.2f}_")
                st.markdown("---")
        else:
            st.warning("❌ Tidak ada hasil yang cocok.")
            saran = suggest_closest_words(query)
            if saran:
                st.info("🔁 Mungkin yang Anda maksud:")
                for s in saran:
                    st.markdown(f"- {s}")
            else:
                st.info("Tidak ditemukan saran kata yang mirip.")
    else:
        st.warning("Masukkan kata kunci terlebih dahulu.")

if col2.button("Lihat Pencarian Populer"):
    st.subheader("📈 Pencarian Terpopuler")
    if search_log:
        for q, count in show_popular_queries(search_log):
            st.write(f"🔍 {q} — {count} kali")
    else:
        st.info("Belum ada histori pencarian.")
