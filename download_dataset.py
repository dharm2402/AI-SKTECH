"""
Script to download Quick Draw dataset for training
"""
import os
import urllib.request
import numpy as np
from PIL import Image

def download_quickdraw_category(category, max_samples=1000):
    """Download a category from Quick Draw dataset"""
    base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
    url = f"{base_url}{category}.npy"
    
    output_dir = "data/sketches"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {category}...")
    
    try:
        # Download the .npy file
        temp_file = f"temp_{category}.npy"
        urllib.request.urlretrieve(url, temp_file)
        
        # Load and save as images
        data = np.load(temp_file)
        
        for i, sketch in enumerate(data[:max_samples]):
            # Reshape from 784 to 28x28
            img_array = sketch.reshape(28, 28)
            
            # Resize to 256x256
            img = Image.fromarray(img_array)
            img = img.resize((256, 256), Image.LANCZOS)
            
            # Save
            img.save(f"{output_dir}/{category}_{i:04d}.png")
        
        # Clean up
        os.remove(temp_file)
        print(f"Downloaded {max_samples} {category} sketches")
        
    except Exception as e:
        print(f"Error downloading {category}: {e}")

if __name__ == "__main__":
    # Popular categories from Quick Draw
    categories = [
        "cat", "dog", "bird", "fish", "horse",
        "car", "airplane", "bicycle", "house", "tree"
    ]
    
    print("Downloading Quick Draw dataset...")
    print("This may take a few minutes...")
    
    for category in categories:
        download_quickdraw_category(category, max_samples=100)
    
    print("\nDataset download complete!")
    print(f"Images saved to: data/sketches/")
