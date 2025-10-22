# Algorithm
An algorithm for sorting reels based on their niches

add cell 
```bash
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA support
!pip install transformers opencv-python pillow tqdm
```

then
```python
from google.colab import drive
drive.mount('/content/drive')
```

then to check if drive is mounted
```bash
%cd /content/drive/MyDrive/
!ls
```

finally paste the python code and change the directory paths accordinly.
```python
# This updated code changes niches to emotion-based categories (e.g., for sentiment analysis in Reels).
# Examples: happy, sad, angry, etc. CLIP can handle text prompts like this for zero-shot classification.
# The rest of the code (checkpointing, CSV/JSON output, /tmp usage) remains unchanged as per your instruction.
# Assumptions: Same as before. Run in Google Colab with CUDA.

# Step 1: Install dependencies (run once)
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA support
# !pip install transformers opencv-python pillow tqdm

# Step 2: Import libraries
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import cv2
import os
from tqdm import tqdm
import pandas as pd
import json  # For checkpointing and JSON output
import shutil

# Check for CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Step 3: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 4: Define emotion-based niches and load CLIP model
niches = ["happy", "sad", "angry", "excited", "calm", "surprised", "fearful", "disgusted", "joyful", "anxious"]  # Customize emotion niches as needed
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

# Step 6: Function to classify a video
def classify_video(video_path, niches, temp_dir="/tmp"):
    frames, temp_paths = extract_frames(video_path, temp_dir=temp_dir)
    if not frames:
        for tp in temp_paths:
            if os.path.exists(tp):
                os.remove(tp)
        return "unknown", 0.0
    
    inputs = processor(text=niches, images=frames[0], return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).squeeze(0)
    predicted_idx = torch.argmax(probs).item()
    confidence = probs[predicted_idx].item()
    
    for tp in temp_paths:
        if os.path.exists(tp):
            os.remove(tp)
    
    return niches[predicted_idx], confidence

# Step 7: Checkpoint functions
checkpoint_file = "/content/drive/MyDrive/reels_checkpoint.json"  # Path to checkpoint
output_csv = "/content/drive/MyDrive/reels_categorized.csv"
output_json = "/content/drive/MyDrive/reels_categorized.json"

def save_checkpoint(last_batch_idx, results_so_far):
    checkpoint_data = {
        "last_batch_idx": last_batch_idx,
        "results": results_so_far
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f)
    print(f"Checkpoint saved at batch {last_batch_idx}")

def load_checkpoint():
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        print(f"Resuming from batch {data['last_batch_idx']}")
        return data['last_batch_idx'], data['results']
    return 0, []  # Start from beginning

# Step 8: Load all video paths from folder
folder_path = "/content/drive/MyDrive/Backup/videos"  # Replace with your folder path
video_extensions = (".mp4", ".avi", ".mov")
video_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(video_extensions)]
print(f"Found {len(video_paths)} videos in {folder_path}")

# Step 9: Load checkpoint and initialize results
start_batch_idx, results = load_checkpoint()
temp_dir = "/tmp"
os.makedirs(temp_dir, exist_ok=True)

# Step 10: Process videos in batches, starting from checkpoint
batch_size = 100
total_batches = (len(video_paths) + batch_size - 1) // batch_size

for i in tqdm(range(start_batch_idx, total_batches), desc="Processing batches", initial=start_batch_idx):
    batch_start = i * batch_size
    batch_end = min((i + 1) * batch_size, len(video_paths))
    batch = video_paths[batch_start:batch_end]
    
    batch_results = []
    for path in batch:
        if os.path.exists(path):
            niche, conf = classify_video(path, niches, temp_dir=temp_dir)
            batch_results.append({"video_path": path, "niche": niche, "confidence": conf})
        else:
            batch_results.append({"video_path": path, "niche": "file_not_found", "confidence": 0.0})
    
    # Append batch results to overall results and save incrementally to CSV
    results.extend(batch_results)
    df_batch = pd.DataFrame(batch_results)
    if i == start_batch_idx and not os.path.exists(output_csv):  # First time, write header
        df_batch.to_csv(output_csv, index=False)
    else:  # Append without header
        df_batch.to_csv(output_csv, mode='a', header=False, index=False)
    
    # Save checkpoint after each batch
    save_checkpoint(i + 1, results)

# Step 11: Final save to both CSV and JSON
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)  # Overwrite final CSV
with open(output_json, 'w') as f:
    json.dump(results, f, indent=4)  # Save as JSON list of dicts
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)  # Remove checkpoint on completion
print(f"Categorization complete. Final results saved to {output_csv} and {output_json}")
print("Temporary files in /tmp have been cleaned up per video.")
```

You can copy paste the code on blackbox or change code to change niches yourself current niches are based on emotions.





