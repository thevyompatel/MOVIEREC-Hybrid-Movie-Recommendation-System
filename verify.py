import os
import sys

# Test script to verify the core logical functions of the Movie Recommender Machine Learning project.

def main():
    print("--- Starting Verification Script ---")
    try:
        from src.preprocessing import load_data, preprocess_data
        from src.content_model import train_content_model, get_content_recommendations
        from src.collaborative_model import create_user_item_matrix, train_collaborative_model, get_collaborative_recommendations
        from src.hybrid_model import get_hybrid_recommendations
        from src.evaluation import evaluate_model
    except ImportError as e:
        print(f"❌ [Import Error] Could not load dependencies: {e}")
        sys.exit(1)

    print("✅ Imports Successful.")

    # 1. Test Dataset Loader & Preprocessing
    try:
        movies_df, ratings_df = load_data('data/movies.csv', 'data/ratings.csv')
        clean_movies, clean_ratings = preprocess_data(movies_df, ratings_df)
        print(f"✅ Data loaded and preprocessed successfully. Movies: {len(clean_movies)}, Ratings: {len(clean_ratings)}")
    except Exception as e:
        print(f"❌ [Data Processing Error] {e}")
        sys.exit(1)

    # 2. Test Content Model
    try:
        tfidf_matrix, content_sim_matrix = train_content_model(clean_movies)
        
        # Test movie: The Matrix (1999) - MovieId: 2571
        matrix_id = 2571
        
        # fallback just in case dataset changed
        if matrix_id not in clean_movies['movieId'].values:
            matrix_id = clean_movies['movieId'].iloc[0]
            
        content_recs = get_content_recommendations(matrix_id, clean_movies, content_sim_matrix, top_n=5)
        print(f"✅ Content Model Trained & Recommendations successfully generated. Returned {len(content_recs)} movies.")
    except Exception as e:
        print(f"❌ [Content Model Error] {e}")
        sys.exit(1)

    # 3. Test Collaborative Model
    try:
        user_item_matrix = create_user_item_matrix(clean_ratings)
        collab_movie_ids_list, collab_sim_matrix = train_collaborative_model(user_item_matrix, n_components=50)
        
        collab_recs = get_collaborative_recommendations(matrix_id, collab_movie_ids_list, collab_sim_matrix, clean_movies, top_n=5)
        print(f"✅ Collaborative Model Trained (Truncated SVD) & Recommendations successfully generated. Returned {len(collab_recs)} movies.")
    except Exception as e:
        print(f"❌ [Collaborative Model Error] {e}")
        sys.exit(1)

    # 4. Test Hybrid Engine Integration
    try:
        hybrid_recs = get_hybrid_recommendations(
            movie_id=matrix_id,
            movies_df=clean_movies,
            content_sim_matrix=content_sim_matrix,
            collab_movie_ids_list=collab_movie_ids_list,
            collab_sim_matrix=collab_sim_matrix,
            weight_content=0.5,
            weight_collab=0.5,
            top_n=5
        )
        print(f"✅ Hybrid Engine successfully dynamically merged models and assigned normalized weights. Returned {len(hybrid_recs)} movies.")
        print(hybrid_recs)
    except Exception as e:
        print(f"❌ [Hybrid Model Error] {e}")
        sys.exit(1)

    # 5. Test Evaluation Metrics
    try:
        # User 1
        eval_metrics = evaluate_model(user_id=1, ratings_df=clean_ratings, hybrid_recommendations=hybrid_recs, k=5)
        print(f"✅ Evaluation metrics generated: {eval_metrics}")
    except Exception as e:
        print(f"❌ [Evaluation Error] {e}")
        sys.exit(1)

    print("\\n🚀 ALL VERIFICATION CHECKS PASSED GLOBALLY. ")

if __name__ == "__main__":
    main()
