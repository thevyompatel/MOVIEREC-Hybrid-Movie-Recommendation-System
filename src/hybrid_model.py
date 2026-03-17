import pandas as pd
from src.content_model import get_content_recommendations
from src.collaborative_model import get_collaborative_recommendations

def get_hybrid_recommendations(movie_id, movies_df, 
                               content_sim_matrix,
                               collab_movie_ids_list, collab_sim_matrix,
                               weight_content=0.5, weight_collab=0.5, top_n=10):
    """
    Combines Content-Based and Collaborative Filtering recommendations.
    Weights are adjustable.
    """
    # Get Content-Based Recommendations
    content_recs = get_content_recommendations(movie_id, movies_df, content_sim_matrix, top_n=top_n * 5)
    
    # Get Collaborative Filtering Recommendations
    collab_recs = get_collaborative_recommendations(movie_id, collab_movie_ids_list, collab_sim_matrix, movies_df, top_n=top_n * 5)
    
    if content_recs.empty and collab_recs.empty:
        return pd.DataFrame()

    # Extract just the scores
    if not content_recs.empty:
        content_df = content_recs[['movieId', 'content_score']].copy()
    else:
        content_df = pd.DataFrame(columns=['movieId', 'content_score'])

    if not collab_recs.empty:
        collab_df = collab_recs[['movieId', 'collab_score']].copy()
    else:
        collab_df = pd.DataFrame(columns=['movieId', 'collab_score'])
        
    # Merge both results on movieId
    hybrid_df = pd.merge(content_df, collab_df, on='movieId', how='outer')
    
    # Fill NA with 0
    hybrid_df['content_score'] = hybrid_df['content_score'].fillna(0)
    hybrid_df['collab_score'] = hybrid_df['collab_score'].fillna(0)
    
    # Normalize scores between 0 and 1 so weights apply fairly
    if hybrid_df['content_score'].max() > 0:
        hybrid_df['content_score'] = hybrid_df['content_score'] / hybrid_df['content_score'].max()
        
    if hybrid_df['collab_score'].max() > 0:
        hybrid_df['collab_score'] = hybrid_df['collab_score'] / hybrid_df['collab_score'].max()
    
    # Calculate final hybrid score
    hybrid_df['final_score'] = (hybrid_df['content_score'] * weight_content) + (hybrid_df['collab_score'] * weight_collab)
    
    # Sort by final score
    hybrid_df = hybrid_df.sort_values(by='final_score', ascending=False)
    
    # Drop duplicates if any and return top n
    hybrid_df = hybrid_df.drop_duplicates(subset=['movieId']).head(top_n)
    
    # Get title and genres back
    final_recs = pd.merge(hybrid_df, movies_df[['movieId', 'title', 'genres']], on='movieId', how='left')
    
    return final_recs[['movieId', 'title', 'genres', 'content_score', 'collab_score', 'final_score']]
