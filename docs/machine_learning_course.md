# Hybrid Movie Recommendation System: Masterclass Course

This document serves as a comprehensive step-by-step guide to understanding the machine learning concepts, mathematics, and architecture behind the Hybrid Movie Recommendation System built in this repository.

---

## SECTION 1: Recommendation System Fundamentals

**1. The Definition**
A recommendation system is an information filtering system that seeks to predict the "rating" or "preference" a user would give to an item. It curates a personalized list of content from a massive catalog to show the user what they are most likely to interact with.

**2. Why It Is Used**
In the age of information overload, choosing what to watch, buy, or read is overwhelming. Recommendation engines solve the "choice paradox," increasing user retention, engagement, and revenue.

**3. Types of Systems**
*   **Content-Based Filtering:** Recommends items similar to those a user liked in the past based on item attributes (e.g., genre, director).
*   **Collaborative Filtering:** Recommends items based on the preferences of similar users (e.g., "Users who liked X also liked Y").
*   **Hybrid Systems:** Combines both methods to overcome individual weaknesses (like the "Cold Start" problem).

**4. Real-World Example**
**Netflix** uses a complex Hybrid System. The homepage rows ("Because you watched *The Matrix*") are often content-based, while "Trending Now" or "Top Picks for You" heavily rely on collaborative filtering based on global user behavior.

---

## SECTION 2: Dataset Understanding

**1. The Definition**
The **MovieLens dataset** is the gold standard for testing recommendation algorithms, created by GroupLens at the University of Minnesota.

**2. Structure**
*   **`movies.csv`**: Contains `movieId`, `title`, and `genres` (e.g., `Action|Adventure|Sci-Fi`). This is our **metadata**.
*   **`ratings.csv`**: Contains `userId`, `movieId`, `rating` (0.5 to 5.0), and `timestamp`. This is our **user-item interaction data**.

**3. Why It Is Used**
It securely provides both explicit feedback (the 1-5 star ratings) and distinct item properties (genres) required to train hybrid models natively.

---

## SECTION 3 & 4: Data Preprocessing & Feature Engineering

**1. The Definition**
Data preprocessing is cleaning raw data, while feature engineering transforms that data into a format a machine learning algorithm can understand mathematically.

**2. Why It Is Used**
Algorithms cannot read "Action|Sci-Fi". They only understand numbers and matrices. Missing values crash models, and improperly formatted text yields garbage predictions.

**3. Code Implementation & Project Fit (`src/preprocessing.py`)**
```python
def preprocess_movies(movies):
    # Handling missing values
    movies['genres'] = movies['genres'].fillna('')
    movies['title'] = movies['title'].fillna('Unknown Title')

    # Feature Extraction: Convert 'Action|Adventure' to 'Action Adventure'
    # This prepares the text for Vectorization
    movies['genres_text'] = movies['genres'].str.replace('|', ' ', regex=False)
    return movies
```

---

## SECTION 5: TF-IDF Vectorization

**1. The Definition**
TF-IDF stands for **Term Frequency-Inverse Document Frequency**. It is a statistical measure that evaluates how relevant a word is to a document in a collection of documents.

**2. Mathematical Intuition**
*   **Term Frequency (TF):** How often a genre appears in a movie. (e.g., "Sci-Fi" appears once in *The Matrix*).
*   **Inverse Document Frequency (IDF):** How common or rare the genre is across *all* movies. If "Drama" is in 80% of movies, it mathematically penalizes the weight of "Drama" so rare genres (like "Film-Noir") stand out more.
*   `TF-IDF Score = TF * IDF`

**3. Why TF-IDF is used**
Instead of just counting words (Count Vectorization), TF-IDF highlights the *unique* characteristics of a movie. 

**4. Code Implementation (`src/content_model.py`)**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
# Transforms text strings into a mathematical matrix of features
tfidf_matrix = tfidf.fit_transform(movies['genres_text']) 
```

---

## SECTION 6: Similarity Measurement

**1. The Definition**
Similarity metrics determine how close two data points (movies) are in a multi-dimensional space.

**2. Mathematical Intuition**
*   **Euclidean Distance:** Measures the straight-line distance between two points. Bad for sparse text data because vectors with more words will appear further apart even if the context is the same.
*   **Cosine Similarity:** Measures the *angle* between two vectors. If the angle is 0° (Cosine = 1), the movies are identical. If the angle is 90° (Cosine = 0), they are completely unrelated. 

**3. Why Cosine Similarity?**
It ignores the magnitude (length) of the vectors. A movie with 10 genres and a movie with 2 genres can still have a massive similarity score if the angles of their vectors align.

---

## SECTION 7: Content-Based Filtering Implementation

**1. The Definition**
We calculate the Cosine Similarity of our TF-IDF genre matrix to find movies that "look" like the selected movie.

**2. Code Implementation (`src/content_model.py`)**
```python
from sklearn.metrics.pairwise import linear_kernel

# linear_kernel is highly optimized math for Cosine Similarity on TF-IDF
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Sorting the array to return the highest similarity scores
sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
```
*How it fits:* When you search "The Matrix", this ranks all 9,000+ movies by how mathematically close their genre matrix is to The Matrix.

---

## SECTION 8: Collaborative Filtering

**1. The Definition**
Filtering based on human behavior rather than content.
*   **User-Based:** "Bob and Alice rate movies similarly. Bob liked *Inception*. Recommend *Inception* to Alice."
*   **Item-Based:** "Users who rate *Inception* highly also rate *Interstellar* highly. Recommend *Interstellar*."

**2. The User-Item Matrix**
We must pivot our data so Rows = Users, Columns = Movies, and Cells = Ratings. Empty cells (movies a user hasn't seen) are filled with `0`.

---

## SECTION 9: Matrix Factorization (SVD)

**1. The Definition & Intuition**
The User-Item matrix is mostly empty (sparse). **Singular Value Decomposition (SVD)** performs **dimensionality reduction**. It compresses the massive matrix into a smaller set of **latent factors** (hidden concepts, like "how much action is in a movie" vs "how much romance").

**2. Code Implementation (`src/collaborative_model.py`)**
```python
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# n_components=50 means we compress thousands of movie dimensions into 50 core hidden traits
svd = TruncatedSVD(n_components=50, random_state=42)
latent_matrix = svd.fit_transform(item_user_matrix)

# We now find item similarity based on these hidden human-behavior traits
item_sim_matrix = cosine_similarity(latent_matrix)
```

---

## SECTION 10: Hybrid Recommendation System

**1. The Definition**
A system that combines the outputs of Content and Collaborative filters.

**2. Why It Is Used**
If a movie is brand new (0 ratings), Collaborative Filtering fails (Cold Start problem). Content-based will still work. Blending them covers all edge cases.

**3. Mathematical Intuition**
We normalize both scores to a 0.0 - 1.0 scale, then apply a weighted sum.
`Final Score = (Content_Score * Weight1) + (Collab_Score * Weight2)`

**4. Code Implementation (`src/hybrid_model.py`)**
```python
# Normalize scores so they are natively equal
hybrid_df['content_score'] = hybrid_df['content_score'] / hybrid_df['content_score'].max()
hybrid_df['collab_score'] = hybrid_df['collab_score'] / hybrid_df['collab_score'].max()

# Apply the mathematical formula
hybrid_df['final_score'] = (hybrid_df['content_score'] * weight_content) + (hybrid_df['collab_score'] * weight_collab)
```

---

## SECTION 11: Evaluation Metrics

**1. The Definition**
How we prove our machine learning model actually works.
*   **RMSE (Root Mean Squared Error):** Measures the average difference between the user's *actual* rating and our *predicted* rating. Lower is better.
*   **Precision@K:** Out of the top K (e.g., 10) recommended movies, what percentage were actually relevant (liked) by the user?
*   **Recall@K:** Out of ALL the movies the user actually liked, what percentage did we successfully capture in our top K recommendations?

---

## SECTION 12 & 13: Web Application & Deployment Concepts

**1. The Definition**
A model sitting in a script is useless to users. **Streamlit** wraps Python data logic into a reactive web interface. 

**2. Code Implementation (`app.py`)**
```python
import streamlit as st
import matplotlib.pyplot as plt

st.title("🎬 Hybrid Movie Recommendation System")
selected_movie = st.selectbox("Type or select a movie:", movie_list)

if st.button("Recommend"):
    recommendations = get_hybrid_recommendations(movie_id, ...)
    st.image(poster_url, use_container_width=True) # Renders TMDB posters natively
```

**3. Deployment Concepts**
*   **`requirements.txt`**: This files tells the server (like AWS, Heroku, or Streamlit Cloud) exactly what dependencies (Pandas, Scikit-learn) it needs to install to run your code.
*   **Hosting:** Deploying turns `localhost:8501` into a public URL accessible worldwide.

---

## SECTION 14: System Design at Netflix/Amazon

**1. Explanation**
At scale, you don't compute similarities on-the-fly. Netflix uses a **Two-Tower Architecture**:
1.  **Candidate Generation:** Extremely fast, lightweight algorithms retrieve 1,000 potential movies from a catalog of millions. 
2.  **Scoring / Ranking:** Heavy, deep neural networks score those 1,000 movies perfectly to pick the top 10. They use real-time caching (Redis) and cluster computing (Apache Spark).

---

## SECTION 15: Advanced Improvements

**1. Where to go next**
*   **Neural Collaborative Filtering (NCF):** Replacing our mathematical SVD dot-product with a Neural Network that can learn non-linear relationships.
*   **Real-Time Recommendations:** Using Kafka streaming to update user vectors the second they pause a movie or click a trailer, adjusting their homepage dynamically within milliseconds.
*   **LLM Embedding Models:** Using tools like OpenAI or BERT to read movie *plot summaries* instead of just genres, creating massive contextual embeddings.
