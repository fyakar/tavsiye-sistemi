import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Veriyi yükleme
file_path = 'CHURN HESAPLAMA.xlsx'  # Excel dosyasının yolu
df = pd.read_excel(file_path)

# Öneri sistemi için gerekli sütunları seçiyoruz
df = df[['Customer_ID', 'Product_Name', 'Sales', 'Category']]

# Ürünlerin etiketlenmesi
label_encoder = LabelEncoder()
df['Product_Name_encoded'] = label_encoder.fit_transform(df['Product_Name'])

# Kullanıcı-Ürün Etkileşim Matrisi oluşturuyoruz
interaction_matrix = df.pivot_table(index='Customer_ID', columns='Product_Name_encoded', values='Sales', aggfunc='sum', fill_value=0)

# Cosine Similarity ile ürünler arasındaki benzerlikleri hesaplıyoruz
cosine_sim = cosine_similarity(interaction_matrix.T)

# Tavsiye Fonksiyonu
def item_based_recommendation(product_name, top_n=5):
    # Ürün adını encode ediyoruz
    product_idx = label_encoder.transform([product_name])[0]
    
    # Ürünler arası benzerlikleri alıyoruz
    similarity_scores = cosine_sim[product_idx]
    
    # En yüksek benzerlik skorlarına sahip N ürünü alıyoruz
    recommended_product_idx = np.argsort(similarity_scores)[::-1][:top_n + 1]  # +1 ekliyoruz çünkü kendisi de dahil olacak
    
    # Kendisi dahil edilmesin, bu yüzden en yüksek benzerlik skoru olan ürünü listeden çıkarıyoruz
    recommended_product_idx = recommended_product_idx[recommended_product_idx != product_idx]
    
    # Ürünlerin isimlerini çözümleyelim
    recommended_products = label_encoder.inverse_transform(recommended_product_idx[:top_n])  # En fazla 'top_n' ürün
    similarity_scores = similarity_scores[recommended_product_idx[:top_n]]  # Benzerlik skorlarını filtreliyoruz
    
    # Tavsiye oranını hesaplıyoruz: (Benzerlik oranı / Maksimum benzerlik skoru) * 100
    max_similarity = similarity_scores[0]  # Maksimum benzerlik
    recommendation_percentages = (similarity_scores / max_similarity) * 100
    
    return recommended_products, recommendation_percentages

# Başlık
st.title("Superstore Ürün Tavsiye Sistemi")

# Sidebar logo ve açıklama
st.sidebar.image('logo.png', use_column_width=True)
st.sidebar.write("Datamigos Ürün Tavsiye Sistemi - Explore products you may like!")

# Sekmeler1
tabs = st.sidebar.radio('Sekmeler:', ['Ürün Tavsiyesi'])

# Sekme 1 - Ürün Tavsiyesi
if tabs == 'Ürün Tavsiyesi':
    st.subheader('Ürün Tavsiyesi')
    
    # Ürün listesini oluşturmak
    product_list = df['Product_Name'].unique().tolist()

    # Ürün ismini almak için hem metin kutusu hem de dropdown listesi
    product_name_input = st.selectbox('Bir ürün seçin:', product_list)  # Dropdown listesi (product_name)
    
    # En fazla kaç ürün önerileceğini seçelim
    top_n_input = st.slider("Önerilecek ürün sayısını seçin", 1, 10, 5)

    # Anında tavsiye alalım (buton olmadan)
    recommendations, recommendation_scores = item_based_recommendation(product_name_input, top_n=top_n_input)
    
    # Tavsiye edilen ürünleri ve tavsiye oranlarını gösterelim
    st.write(f"{product_name_input} ürününü alanlar şunları da alabilir:")
    for i, (product, score) in enumerate(zip(recommendations, recommendation_scores), 1):
        st.write(f"{i}. {product} - Tavsiye Oranı: {score:.2f}%")
