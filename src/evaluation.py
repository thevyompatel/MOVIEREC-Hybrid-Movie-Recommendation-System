import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE).
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def precision_at_k(recommended_items, relevant_items, k=10):
    """
    Calculate Precision@K: Proportion of recommended items in the top-k set that are relevant.
    """
    recommended_at_k = recommended_items[:k]
    relevant_and_recommended = set(recommended_at_k).intersection(set(relevant_items))
    return len(relevant_and_recommended) / k if k > 0 else 0.0

def recall_at_k(recommended_items, relevant_items, k=10):
    """
    Calculate Recall@K: Proportion of relevant items that are found in the top-k recommendations.
    """
    recommended_at_k = recommended_items[:k]
    relevant_and_recommended = set(recommended_at_k).intersection(set(relevant_items))
    return len(relevant_and_recommended) / len(relevant_items) if len(relevant_items) > 0 else 0.0

def evaluate_model(user_id, ratings_df, hybrid_recommendations, k=10):
    """
    Evaluates the model for a specific user.
    We consider a relevant item as anything the user rated >= 4.0.
    """
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    
    # Target relevant items (Rated 4 and above)
    relevant_items = user_ratings[user_ratings['rating'] >= 4.0]['movieId'].tolist()
    
    # Recommendations given by the system
    recommended_items = hybrid_recommendations['movieId'].tolist()
    
    p_at_k = precision_at_k(recommended_items, relevant_items, k)
    r_at_k = recall_at_k(recommended_items, relevant_items, k)
    
    return {
        'Precision@K': p_at_k,
        'Recall@K': r_at_k
    }
