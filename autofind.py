# This is the full updated script for unsupervised niche discovery on 10k Instagram Reels.
# It uses CLIP embeddings + K-Means clustering to find niches on its own (no predefined categories).
# Niches are discovered as clusters (e.g., "niche_0", "niche_1"); inspect and label them manually.
# Features: Checkpointing for reload/restart, /tmp for temp files, CSV/JSON output (saved only at end), auto-detection of num_clusters.
# Run in Google Colab with CUDA enabled.

# Step 1: Install dependencies (run once)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA support
!pip install transformers opencv-python pillow tqdm scikit-learn  # Added sklearn for clustering

# Step 2: Import libraries
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import cv2
import os
from tqdm import tqdm
import pandas as pd
import json
from sklearn.cluster import KMeans  # For clustering
from sklearn.metrics import silhouette_score  # For auto-detecting clusters
import numpy as np

# Check for CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Step 3: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 4: Load CLIP model (no predefined niches needed)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Step 5: Function to extract frames (save to /tmp temporarily)
def extract_frames(video_path, num_frames=5, temp_dir="/tmp"):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []
    step = max(1, total_frames // num_frames)
    temp_paths = []
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            temp_path = os.path.join(temp_dir, f"frame_{os.path.basename(video_path)}_{i}.jpg")
            img.save(temp_path)
            temp_paths.append(temp_path)
            frames.append(img)
        if len(frames) >= num_frames:
            break
    cap.release()
    return frames, temp_paths

# Step 6: Function to get CLIP embedding for a video
def get_video_embedding(video_path, temp_dir="/tmp"):
    frames, temp_paths = extract_frames(video_path, temp_dir=temp_dir)
    if not frames:
        for tp in temp_paths:
            if os.path.exists(tp):
                os.remove(tp)
        return None
    
    # Get embedding for the first frame
    inputs = processor(images=frames[0], return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs).cpu().numpy().flatten()
    
    for tp in temp_paths:
        if os.path.exists(tp):
            os.remove(tp)
    
    return embedding

# Step 7: Checkpoint functions (updated for embeddings)
checkpoint_file = "/content/drive/MyDrive/reels_checkpoint.json"
output_csv = "/content/drive/MyDrive/reels_categorized.csv"
output_json = "/content/drive/MyDrive/reels_categorized.json"

def save_checkpoint(last_batch_idx, embeddings_so_far, paths_so_far):
    checkpoint_data = {
        "last_batch_idx": last_batch_idx,
        "embeddings": embeddings_so_far.tolist() if isinstance(embeddings_so_far, np.ndarray) else embeddings_so_far,
        "paths": paths_so_far
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f)
    print(f"Checkpoint saved at batch {last_batch_idx}")

def load_checkpoint():
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        print(f"Resuming from batch {data['last_batch_idx']}")
        return data['last_batch_idx'], np.array(data['embeddings']), data['paths']
    return 0, np.array([]), []

# Step 8: Load all video paths from folder
folder_path = "/content/drive/MyDrive/reels/"  # Replace with your folder path
video_extensions = (".mp4", ".avi", ".mov")
video_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(video_extensions)]
print(f"Found {len(video_paths)} videos in {folder_path}")

# Step 9: Load checkpoint and initialize
start_batch_idx, all_embeddings, processed_paths = load_checkpoint()
temp_dir = "/tmp"
os.makedirs(temp_dir, exist_ok=True)

# Step 10: Extract embeddings in batches
batch_size = 100
total_batches = (len(video_paths) + batch_size - 1) // batch_size

for i in tqdm(range(start_batch_idx, total_batches), desc="Extracting embeddings", initial=start_batch_idx):
    batch_start = i * batch_size
    batch_end = min((i + 1) * batch_size, len(video_paths))
    batch_paths = video_paths[batch_start:batch_end]
    
    batch_embeddings = []
    valid_paths = []
    for path in batch_paths:
        if os.path.exists(path):
            emb = get_video_embedding(path, temp_dir=temp_dir)
            if emb is not None:
                batch_embeddings.append(emb)
                valid_paths.append(path)
    
    # Accumulate embeddings and paths
    if batch_embeddings:
        all_embeddings = np.vstack([all_embeddings, np.array(batch_embeddings)]) if all_embeddings.size else np.array(batch_embeddings)
        processed_paths.extend(valid_paths)
    
    # Save checkpoint
    save_checkpoint(i + 1, all_embeddings, processed_paths)

# Step 11: Perform clustering to discover niches (with auto-detection of num_clusters)
if all_embeddings.size == 0:
    print("No embeddings extracted. Check videos.")
else:
    # Auto-detect optimal clusters (range 2-20; adjust as needed)
    scores = []
    for k in range(2, 21):  # Test 2 to 20 clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(all_embeddings)
        score = silhouette_score(all_embeddings, labels)
        scores.append(score)
    
    best_k = scores.index(max(scores)) + 2  # Best k based on highest silhouette score
    print(f"Optimal number of clusters: {best_k} (silhouette score: {max(scores):.2f})")
    
    # Use best_k for clustering
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    cluster_labels = kmeans.fit_predict(all_embeddings)
    
    # Create results
    results = []
    for path, label in zip(processed_paths, cluster_labels):
        results.append({"video_path": path, "niche": f"niche_{label}", "confidence": 1.0})  # Confidence as 1.0 for clustering
    
    # Save to CSV and JSON (only at the end)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Clustering complete. Found {best_k} niches. Results saved to {output_csv} and {output_json}")

if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)
print("Temporary files in /tmp have been cleaned up.")
