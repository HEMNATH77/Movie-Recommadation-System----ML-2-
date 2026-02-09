import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Spotify Movie Recommender", layout="wide")

# ---------------------------------------------------
# SPOTIFY THEME CSS
# ---------------------------------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #121212;
    color: white;
}

/* Hide Streamlit menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Main Title */
.main-title {
    font-size: 60px;
    font-weight: 800;
    text-align: center;
    color: #1DB954;
    margin-bottom: 40px;
}

/* Section */
.section-title {
    font-size: 26px;
    font-weight: 600;
    margin-bottom: 20px;
}

/* Input */
input {
    background-color: #181818 !important;
    color: white !important;
    border-radius: 30px !important;
}

/* Card */
.card {
    background-color: #181818;
    padding: 25px;
    border-radius: 18px;
    margin: 15px;
    transition: 0.3s ease;
    border: 1px solid #282828;
}

.card:hover {
    background-color: #242424;
    transform: scale(1.04);
    border: 1px solid #1DB954;
}

/* Buttons */
div.stButton > button {
    background-color: #1DB954;
    color: black;
    font-weight: bold;
    border-radius: 30px;
    height: 45px;
    border: none;
}

div.stButton > button:hover {
    background-color: #17a74a;
    transform: scale(1.05);
}

/* Slider */
.css-1cpxqw2 {
    color: #1DB954 !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Movie Recommendation System</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")

    df['writers'] = df['writers'].fillna('Unknown')
    df['genres'] = df['genres'].fillna('Unknown')
    df['directors'] = df['directors'].fillna('Unknown')

    df['combined'] = df['genres'] + " " + df['directors'] + " " + df['writers']
    return df

df = load_data()

# ---------------------------------------------------
# CREATE SIMILARITY MATRIX
# ---------------------------------------------------
@st.cache_resource
def create_similarity(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(data['combined'])
    return cosine_similarity(matrix)

similarity_matrix = create_similarity(df)

# ---------------------------------------------------
# RECOMMEND FUNCTION
# ---------------------------------------------------
def recommend(movie_name, top_n=6):

    movie_match = df[df['primaryTitle'].str.lower().str.contains(movie_name.lower())]

    if movie_match.empty:
        return None

    idx = movie_match.index[0]
    scores = similarity_matrix[idx]
    top_indices = np.argsort(scores)[::-1][1:top_n+1]

    return df.iloc[top_indices]

# ---------------------------------------------------
# USER INPUT SECTION
# ---------------------------------------------------
st.markdown('<div class="section-title">🔍 Search Your Movie</div>', unsafe_allow_html=True)

user_input = st.text_input("Type movie name")

top_n = st.slider("Number of Recommendations", 1, 10, 6)

col1, col2 = st.columns(2)

with col1:
    recommend_btn = st.button("🎯 Recommend", use_container_width=True)

with col2:
    surprise_btn = st.button("🎲 Surprise Me", use_container_width=True)

# ---------------------------------------------------
# BUTTON ACTIONS
# ---------------------------------------------------

if surprise_btn:
    random_movie = random.choice(df['primaryTitle'].values)
    st.success(f"🎉 Surprise Pick: {random_movie}")
    user_input = random_movie
    recommend_btn = True

if recommend_btn:

    if user_input == "":
        st.warning("Please type a movie name first 😅")
    else:
        with st.spinner("Curating your playlist... 🎬"):
            recommendations = recommend(user_input, top_n)

        if recommendations is None:
            st.error("Movie not found. Try another name!")
        else:
            st.markdown('<div class="section-title">🔥 Recommended For You</div>', unsafe_allow_html=True)

            cols = st.columns(3)

            for i in range(len(recommendations)):
                movie = recommendations.iloc[i]

                rating = movie['averageRating']
                stars = "⭐" * int(round(rating / 2))

                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="card">
                        <h3 style="color:#1DB954;">{movie['primaryTitle']}</h3>
                        <p>{stars}</p>
                        <p>🎭 {movie['genres']}</p>
                        <p>🎬 {movie['directors']}</p>
                        <p>📅 {movie['startYear']}</p>
                        <p>⏳ {movie['runtimeMinutes']} mins</p>
                    </div>
                    """, unsafe_allow_html=True)
