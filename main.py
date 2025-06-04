import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import ipywidgets as widgets
from IPython.display import display, clear_output

import streamlit as st

# Load dataset dari file CSV
df = pd.read_csv('content/menu_kopi.csv')
# Konversi DataFrame ke list of dicts
menu = df.to_dict(orient='records')

# Persiapan corpus
corpus = [
    f"{item['nama']} {item['kategori']} {item['deskripsi']}" for item in menu]

# TF-IDF vectorizer dan matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# Log pencarian
search_log = []


def search_menu(query, kategori_filter=None):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    results = []
    for idx, score in enumerate(scores):
        item = menu[idx]
        if kategori_filter and kategori_filter.lower() != 'semua' and item["kategori"].lower() != kategori_filter.lower():
            continue
        if score > 0:
            results.append((item, score))

    results.sort(key=lambda x: -x[1])
    return results


def show_popular_queries(top_n=5):
    if not search_log:
        print("Belum ada histori pencarian.")
        return
    print("\n📊 Pencarian Terpopuler:")
    for q, c in Counter(search_log).most_common(top_n):
        print(f"🔍 {q} — {c} kali")


# Widgets interaktif
query_input = widgets.Text(
    value='', placeholder='Masukkan kata kunci...', description='Kata kunci:')
kategori_dropdown = widgets.Dropdown(
    options=['Semua', 'Hot', 'Cold'], description='Kategori:')
search_button = widgets.Button(description="🔍 Cari")
popular_button = widgets.Button(description="📈 Lihat Populer")
output = widgets.Output()


def on_search_clicked(b):
    with output:
        clear_output()
        query = query_input.value.strip()
        kategori = kategori_dropdown.value
        if not query:
            print("Masukkan kata kunci terlebih dahulu.")
            return

        search_log.append(query)
        results = search_menu(query, kategori_filter=kategori)

        if results:
            print(f"Hasil pencarian untuk: '{query}' (Kategori: {kategori})\n")
            for item, score in results:
                print(f"- {item['nama']} ({item['kategori']})")
                print(f"  Deskripsi: {item['deskripsi']}")
                print(f"  Skor: {score:.2f}\n")
        else:
            print("Tidak ada hasil yang cocok.")


def on_popular_clicked(b):
    with output:
        clear_output()
        show_popular_queries()


search_button.on_click(on_search_clicked)
popular_button.on_click(on_popular_clicked)

display(query_input, kategori_dropdown, search_button, popular_button, output)
