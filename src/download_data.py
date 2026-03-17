import os
import urllib.request
import zipfile
import ssl
import socket

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
    
    # Increase timeout and set up SSL context
    socket.setdefaulttimeout(300)  # 5 minute timeout
    ssl._create_default_https_context = ssl._create_unverified_context
    
    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=_download_progress)
        print("\nDownload completed!")
    except Exception as e:
        print(f"Failed to download dataset. Error: {e}")
        raise
        
    try:
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
    except Exception as e:
        print(f"Error during extraction: {e}")
        # Clean up partial downloads
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise

def _download_progress(block_num, block_size, total_size):
    """Display download progress"""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(downloaded * 100 / total_size, 100)
        print(f"\rDownloading... {percent:.1f}%", end="")

if __name__ == "__main__":
    download_and_extract_data()
