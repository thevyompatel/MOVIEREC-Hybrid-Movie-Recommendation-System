import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import urllib.parse
import os

from src.preprocessing import load_data, preprocess_data
from src.content_model import train_content_model
from src.collaborative_model import create_user_item_matrix, train_collaborative_model
from src.hybrid_model import get_hybrid_recommendations
from src.download_data import download_and_extract_data

# Cache data loading so it's not run on every interaction
@st.cache_data
def load_and_prepare_data():
    try:
        if not os.path.exists('data/movies.csv') or not os.path.exists('data/ratings.csv'):
            st.info("📥 Downloading MovieLens dataset on first run... (This may take 1-2 minutes)")
            # Automatically download dataset if not found (e.g. deployed on Streamlit Cloud)
            download_and_extract_data()
            
            # Double check if download succeeded
            if not os.path.exists('data/movies.csv'):
                st.error("Dataset download failed. Please check your internet connection.")
                return None, None, None, None, None, None, None
            
        movies, ratings = load_data('data/movies.csv', 'data/ratings.csv')
        movies_clean, ratings_clean = preprocess_data(movies, ratings)
        
        # Precompute models
        # Content-based
        tfidf_matrix, content_sim_matrix = train_content_model(movies_clean)
        
        # Collaborative
        user_item_matrix = create_user_item_matrix(ratings_clean)
        collab_movie_ids_list, collab_sim_matrix = train_collaborative_model(user_item_matrix, n_components=50)
        
        return movies_clean, ratings_clean, content_sim_matrix, collab_movie_ids_list, collab_sim_matrix, tfidf_matrix, user_item_matrix
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None, None, None, None, None, None, None

# Function to fetch movie poster
@st.cache_data
def fetch_poster(movie_title, api_key=""):
    # If no api key is provided, we can maybe query TMDB by its search API 
    if not api_key:
        return "https://via.placeholder.com/300x450.png?text=No+Poster"
        
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={urllib.parse.quote(movie_title)}"
    try:
        data = requests.get(url).json()
        if data['results']:
            poster_path = data['results'][0]['poster_path']
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        return "https://via.placeholder.com/300x450.png?text=Error"
        
    return "https://via.placeholder.com/300x450.png?text=No+Poster"

def main():
    st.set_page_config(page_title="Movie Recommender System", layout="wide")
    
    st.title("🎬 Hybrid Movie Recommendation System")
    st.markdown("This project combines Content-Based Filtering & Collaborative Filtering to provide personalized movie recommendations.")

    # Sidebar for Config
    st.sidebar.header("⚙️ Configuration")
    tmdb_api_key = st.sidebar.text_input("TMDB API Key (Optional)", type="password", help="Enter TMDB API Key for movie posters")
    
    st.sidebar.subheader("Hybrid Model Weights")
    weight_content = st.sidebar.slider("Content-Based Weight", 0.0, 1.0, 0.5, 0.1)
    weight_collab = st.sidebar.slider("Collaborative Weight", 0.0, 1.0, 0.5, 0.1)
    
    # Check weights sum
    if weight_content + weight_collab == 0:
        st.sidebar.warning("Sum of weights must be > 0. Reverting to 0.5/0.5")
        weight_content = 0.5
        weight_collab = 0.5

    # Display Loading Message
    with st.spinner("Loading and processing data..."):
        movies_df, ratings_df, content_sim, collab_movie_ids, collab_sim, _, _ = load_and_prepare_data()
        
    if movies_df is None:
        st.error("❌ Could not load the dataset. Please try reloading the page in a moment. If the problem persists, the dataset download may have failed on the server.")
        st.stop()
        
    # Main Tabs
    tab1, tab2 = st.tabs(["🔍 Recommend Movies", "📊 Data Insights"])
    
    with tab1:
        st.subheader("Find Similar Movies")
        
        # Movie Search Bar / Dropdown
        movie_list = movies_df['title'].tolist()
        selected_movie = st.selectbox("Type or select a movie you like:", movie_list)
        
        if st.button("Recommend", type="primary"):
            # Get movieId of selected movie
            movie_id = movies_df[movies_df['title'] == selected_movie]['movieId'].values[0]
            
            with st.spinner(f"Generating recommendations for '{selected_movie}'..."):
                recommendations = get_hybrid_recommendations(
                    movie_id=movie_id,
                    movies_df=movies_df,
                    content_sim_matrix=content_sim,
                    collab_movie_ids_list=collab_movie_ids,
                    collab_sim_matrix=collab_sim,
                    weight_content=weight_content,
                    weight_collab=weight_collab,
                    top_n=10
                )
                
            if recommendations.empty:
                st.warning("Could not find recommendations for this movie.")
            else:
                st.success("Here are your Top 10 Recommendations!")
                
                # We use container/columns to display
                cols = st.columns(5)
                for index, row in recommendations.iterrows():
                    col_idx = (index) % 5
                    with cols[col_idx]:
                        # Try to remove year from title for better search on TMDB
                        search_title = row['title'].split('(')[0].strip() if '(' in row['title'] else row['title']
                        poster_url = fetch_poster(search_title, tmdb_api_key)
                        
                        st.image(poster_url, use_container_width=True)
                        st.markdown(f"**{row['title']}**")
                        st.caption(f"Genres: {row['genres']}")
                        st.progress(min(row['final_score'], 1.0))
                        st.caption(f"Score:  {row['final_score']:.2f}")

    with tab2:
        st.header("Dataset Insights Visualizations")
        
        st.subheader("1. Genre Distribution")
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        # Count genres
        genre_series = movies_df['genres'].str.split('|').explode()
        genre_count = genre_series.value_counts()
        sns.barplot(y=genre_count.index, x=genre_count.values, palette='viridis', ax=ax1)
        ax1.set_xlabel('Number of Movies')
        ax1.set_ylabel('Genre')
        ax1.set_title('Movies per Genre')
        st.pyplot(fig1)
        
        st.subheader("2. Ratings Distribution")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.histplot(ratings_df['rating'], bins=10, kde=True, ax=ax2, color='coral')
        ax2.set_xlabel('Rating')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of User Ratings')
        st.pyplot(fig2)
        
        st.subheader("3. Most Popular Movies (By Rating Count)")
        popular_movies = ratings_df.groupby('movieId').size().reset_index(name='rating_count')
        popular_movies = popular_movies.sort_values('rating_count', ascending=False).head(10)
        popular_movies = popular_movies.merge(movies_df[['movieId', 'title']], on='movieId')
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.barplot(x='rating_count', y='title', data=popular_movies, palette='magma', ax=ax3)
        ax3.set_xlabel('Number of Ratings')
        ax3.set_ylabel('Movie Title')
        ax3.set_title('Top 10 Most Rated Movies')
        st.pyplot(fig3)

if __name__ == '__main__':
    main()
