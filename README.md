# Movie Recommendation System


A **content-based movie recommendation system** built using **Python, Machine Learning, and Streamlit**, designed with a **Spotify-inspired premium UI**.  
Type a movie name and instantly discover similar movies based on genres, directors, and writers.

---

## 🚀 Live Features

✅ Type any movie name (auto-matching)  
✅ Get similar movie recommendations instantly  
✅ Spotify-style dark premium UI  
✅ Surprise Me 🎲 button for random movie discovery  
✅ Fast recommendations using TF-IDF + Cosine Similarity  
✅ Clean, interactive, portfolio-ready design  

---

## 🧠 How It Works

This project uses **Content-Based Filtering**:

1. Combine movie features:
   - Genres  
   - Directors  
   - Writers  

2. Convert text data into numerical vectors using **TF-IDF Vectorizer**

3. Compute similarity between movies using **Cosine Similarity**

4. Recommend movies with the highest similarity scores

---

## 🛠️ Technologies Used

- **Python**
- **Streamlit** (Web App Framework)
- **Pandas & NumPy** (Data Handling)
- **Scikit-Learn**
  - TF-IDF Vectorizer
  - Cosine Similarity
- **Custom CSS** (Spotify-style UI)

---

## 📂 Dataset Columns Used

- `primaryTitle`
- `genres`
- `directors`
- `writers`
- `averageRating`
- `startYear`
- `runtimeMinutes`

Missing values are handled using **safe null imputation**.
