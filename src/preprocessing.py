import pandas as pd
import numpy as np

def load_data(movies_path='data/movies.csv', ratings_path='data/ratings.csv'):
    """
    Load movies and ratings datasets.
    """
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings

def preprocess_movies(movies):
    """
    Preprocess the movies dataset.
    - Fills missing values.
    - Formats genres to space-separated text for vectorization.
    """
    # Fill missing values if any
    movies['genres'] = movies['genres'].fillna('')
    movies['title'] = movies['title'].fillna('Unknown Title')

    # Convert genres format from 'Action|Adventure|Sci-Fi' to 'Action Adventure Sci-Fi'
    movies['genres_text'] = movies['genres'].str.replace('|', ' ', regex=False)
    
    return movies

def preprocess_data(movies, ratings):
    """
    Preprocess and merge movies and ratings if necessary.
    Returns cleaned individual datasets.
    """
    movies_clean = preprocess_movies(movies)
    
    # Drop rows without a UserId or MovieId
    ratings_clean = ratings.dropna(subset=['userId', 'movieId', 'rating'])
    
    return movies_clean, ratings_clean
