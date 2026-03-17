import os
import urllib.request
import zipfile

def download_and_extract_data(data_dir='data'):
    print("Checking for MovieLens dataset...")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    movies_path = os.path.join(data_dir, 'movies.csv')
    ratings_path = os.path.join(data_dir, 'ratings.csv')
    
    if os.path.exists(movies_path) and os.path.exists(ratings_path):
        print("Dataset already exists!")
        return

    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    zip_path = os.path.join(data_dir, "ml-latest-small.zip")
    
    print(f"Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"Failed to download dataset. Using fallback URL. Error: {e}")
        # Sometimes there are SSL cert issues depending on the python environment
        # We could use requests to ignore ssl if needed, but urlretrieve is simple
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(url, zip_path)
        
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
        
    # Move files to main data directory
    extracted_dir = os.path.join(data_dir, "ml-latest-small")
    
    for file in os.listdir(extracted_dir):
        if file.endswith('.csv'):
            # Replace if already exists (safe move)
            source = os.path.join(extracted_dir, file)
            destination = os.path.join(data_dir, file)
            if os.path.exists(destination):
                os.remove(destination)
            os.rename(source, destination)
            
    # Cleanup
    print("Cleaning up temporary files...")
    os.remove(zip_path)
    import shutil
    shutil.rmtree(extracted_dir)
    print("Dataset ready in 'data/' directory.")

if __name__ == "__main__":
    download_and_extract_data()
