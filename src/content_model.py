import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def train_content_model(movies):
    """
    Builds the content-based recommendation system using TF-IDF on 'genres_text'.
    Returns only the TF-IDF matrix to avoid storing a huge NxN similarity matrix.
    """
    # Instantiate TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Fit and transform the genres data to create the TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(movies['genres_text'])
    
    return tfidf_matrix

def get_content_recommendations(movie_id, movies, tfidf_matrix, top_n=10):
    """
    Get n most similar movies based on content (genres).
    Returns a dataframe of the top recommended movies.
    """
    # Create a mapping of movie_id to index
    indices = pd.Series(movies.index, index=movies['movieId']).drop_duplicates()
    
    if movie_id not in indices:
        return pd.DataFrame() # Return empty if movie is not found
        
    idx = indices[movie_id]
    
    # Compute similarity only for selected movie vs all movies (memory efficient)
    sim_values = linear_kernel(tfidf_matrix[idx:idx + 1], tfidf_matrix).flatten()
    sim_scores = list(enumerate(sim_values))
    
    # Sort the movies based on similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Exclude the movie itself (the 0th element in the sorted list)
    sim_scores = sim_scores[1:top_n+1]
    
    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    
    rec_movies = movies.iloc[movie_indices].copy()
    rec_movies['content_score'] = scores
    
    return rec_movies[['movieId', 'title', 'genres', 'content_score']]
