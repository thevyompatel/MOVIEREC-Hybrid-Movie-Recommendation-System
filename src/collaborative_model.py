import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

def create_user_item_matrix(ratings):
    """
    Converts ratings into a user-item matrix where:
    Rows = userIds
    Columns = movieIds
    Values = ratings
    Missing values (unrated movies) are filled with 0.
    """
    # Pivot the dataset to create the user-item matrix
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    return user_item_matrix

def train_collaborative_model(user_item_matrix, n_components=50):
    """
    Trains an item-based collaborative filtering model using TruncatedSVD.
    This factorizes the transposed user-item matrix (items are rows) to discover latent features.
    """
    # Items as rows, users as columns
    item_user_matrix = user_item_matrix.T
    
    # We use TruncatedSVD to decompose the sparse matrix
    # n_components determines the number of latent factors
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    latent_matrix = svd.fit_transform(item_user_matrix)
    
    # Calculate similarity between all items/movies using their latent representations
    item_sim_matrix = cosine_similarity(latent_matrix)
    
    return item_user_matrix.index.values, item_sim_matrix

def get_collaborative_recommendations(movie_id, movie_ids_list, item_sim_matrix, movies_df, top_n=10):
    """
    Provides movie recommendations based on collaborative filtering item-to-item similarity.
    """
    # Find the index of the movie id in our list
    if movie_id not in movie_ids_list:
        return pd.DataFrame()
        
    idx = list(movie_ids_list).index(movie_id)
    
    # Get similarity scores
    sim_scores = list(enumerate(item_sim_matrix[idx]))
    
    # Sort them in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top n most similar movies (skipping the first one which is itself)
    sim_scores = sim_scores[1:top_n+1]
    
    # Get the movie indices in the matrix context (which corresponds to movieIds)
    rec_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    
    rec_movie_ids = [movie_ids_list[i] for i in rec_indices]
    
    # Build dataframe to return
    rec_df = pd.DataFrame({
        'movieId': rec_movie_ids,
        'collab_score': scores
    })
    
    # Merge with original movies df to get titles
    result = rec_df.merge(movies_df[['movieId', 'title', 'genres']], on='movieId', how='left')
    
    return result
