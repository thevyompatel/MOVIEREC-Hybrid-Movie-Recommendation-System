<div align=\"center\">
  <h1>🎬 Hybrid Movie Recommendation System</h1>
  <p>
    <b>A Production-Ready Machine Learning Web Application combining Content-Based and Collaborative Filtering using the MovieLens Dataset.</b>
  </p>
  
  [![Python version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
  [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
  [![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
</div>

---

## 📖 Overview

The **Hybrid Movie Recommendation System** is an end-to-end Machine Learning pipeline that actively curates and suggests top movie recommendations. By dynamically merging **Content-Based Filtering** (TF-IDF vectorization on movie genres) and **Collaborative Filtering** (Matrix Factorization via Truncated SVD on user ratings), this engine effectively balances user behavior with inherent item characteristics.

This project is fully designed for scale, structured like an enterprise-level ML architecture, and deployed via a dynamic **Streamlit Cloud** front-end interface.

### ✨ Key Features
- **Hybrid Fusion Engine:** Intelligently combats the "Cold Start" problem by normalizing and weighting both Content and Collaborative scores.
- **Dynamic Dataset Pipelines:** The system is configured to auto-download and verify the `ml-latest-small` GroupLens datasets on boot, avoiding GitHub file size limits.
- **TMDB API Integration:** Live poster fetching via the official TMDB search API endpoints natively rendered in the app.
- **Visual Analytics:** Real-time Exploratory Data Analysis (EDA) dashboard displaying genre densities and ratings distributions via `seaborn` and `matplotlib`.
- **Customizable ML Parameters:** Adjust model weights in real-time between `0.0` to `1.0` through the UI to test mathematical outcomes natively.

---

## 🛠️ Architecture & Tech Stack

*   **Language:** Python 3.10+
*   **Web Framework:** Streamlit
*   **Data Processing:** Pandas, NumPy
*   **Machine Learning Model:** Scikit-Learn (`TfidfVectorizer`, `TruncatedSVD`, `linear_kernel`, `cosine_similarity`)
*   **API Calls:** Requests, urllib3
*   **Visualization:** Matplotlib, Seaborn

---

## 🚀 Live Demo & Deployment

1. 🌐 Streamlit Community Cloud (Recommended)

You already designed for this — fastest path.

🔗 Deploy here:

👉 https://share.streamlit.io/

✅ What you’ll get

Public URL like:
https://your-username-movie-recommender-app-xyz.streamlit.app

⚡ Steps (optimized)

Push your repo to GitHub

Go to Streamlit Cloud

Select:

Repo: movie-recommender

Branch: main

File: app.py

Add Secrets (for TMDB API):

TMDB_API_KEY=your_api_key_here

Click Deploy

💡 Example Live Demo (structure)
https://movie-recommender-hybrid.streamlit.app
---

## 💻 Local Installation & Usage

You can safely run the models and the entire Streamlit dashboard locally.

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/movie-recommender.git
cd movie-recommender
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py
```
*Note: Upon running the application for the first time, it will automatically download the 1MB MovieLens `.zip`, extract it to the `/data/` folder, and run initial model pre-caching. This may take ~10 seconds before boot.*

---

## 📁 Project Structure

```bash
movie-recommender/
│
├── data/                    # (Git-ignored) Directory for raw/cleaned downloaded data
├── docs/                    # Contains detailed ML Masterclass Course documentation
│   └── machine_learning_course.md
│
├── src/                     # Core Machine Learning Framework Modules
│   ├── download_data.py     # Automates MovieLens dataset fetching
│   ├── preprocessing.py     # Cleans missing values and generates text features
│   ├── content_model.py     # Generates TF-IDF and Cosine Similarity matrices
│   ├── collaborative_model.py # Generates User-Item Matrix & Truncated SVD models
│   ├── hybrid_model.py      # Combines multi-model scores with weight equations
│   └── evaluation.py        # Validations (RMSE, Precision@K, Recall@K)
│
├── notebooks/               # Standalone Data Research Modules
│   └── EDA.ipynb            
│
├── app.py                   # Primary Streamlit Application Entrypoint
├── verify.py                # Command-Line Integration Testing Script
├── requirements.txt         # Production Dependency Tracking
└── README.md                # Project Documentation
```

---

## 🧠 Core Machine Learning Concepts

*   **Content-Based Filtering:** Analyzes the `genres` text natively formatted as (e.g., `Action Adventure Sci-Fi`), computes document term frequencies against inverse document populations (TF-IDF), and ranks nearest mathematically identical movie properties vectors (Cosine Similarity).
*   **Collaborative Filtering:** Identifies behavioral overlap natively among users. Transposes `movies.csv` and `ratings.csv` into a sparse memory matrix and solves its dimensionality reduction via Matrix Factorization (`TruncatedSVD`). It predicts "Users who like **X**, generally behave like similar users who enjoy **Y**".

*Read the comprehensive [Machine Learning Masterclass Guide](docs/machine_learning_course.md) included in this repository to understand the absolute granular mathematics and logic driving this recommendation framework.*

---

## 📊 Dataset Attribution

This software utilizes the heavily researched **MovieLens Dataset** (Specifically the `ml-latest-small` node).

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872>

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
If you'd like to support the build, give it a ⭐️!
