import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def train_content_model(movies):
    """
    Builds the content-based recommendation system using TF-IDF on 'genres_text'.
    Returns the TF-IDF matrix and the cosine similarity matrix.
    """
    # Instantiate TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Fit and transform the genres data to create the TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(movies['genres_text'])
    
    # Compute the cosine similarity matrix using linear_kernel (which is equivalent to cosine similarity for tfidf)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    return tfidf_matrix, cosine_sim

def get_content_recommendations(movie_id, movies, cosine_sim, top_n=10):
    """
    Get n most similar movies based on content (genres).
    Returns a dataframe of the top recommended movies.
    """
    # Create a mapping of movie_id to index
    indices = pd.Series(movies.index, index=movies['movieId']).drop_duplicates()
    
    if movie_id not in indices:
        return pd.DataFrame() # Return empty if movie is not found
        
    idx = indices[movie_id]
    
    # Get pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Exclude the movie itself (the 0th element in the sorted list)
    sim_scores = sim_scores[1:top_n+1]
    
    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    
    rec_movies = movies.iloc[movie_indices].copy()
    rec_movies['content_score'] = scores
    
    return rec_movies[['movieId', 'title', 'genres', 'content_score']]
