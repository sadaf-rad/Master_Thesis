import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# === Set custom Hugging Face cache directory to avoid home quota ===
os.environ['TRANSFORMERS_CACHE'] = '/scratch/s3986160/huggingface'
os.environ['HF_HOME'] = '/scratch/s3986160/huggingface'

# === Step 1: Load user descriptions ===
print("[1] Loading user descriptions...")
df = pd.read_csv("/home/s3986160/master-thesis/Improvements/user_descriptions_2020.csv")
descriptions = df['user_description'].tolist()

# === Step 2: Load MiniLM model ===
print("[2] Loading MiniLM...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# === Step 3: Generate embeddings ===
print("[3] Generating embeddings...")
embeddings = model.encode(descriptions, show_progress_bar=True)

# === Step 4: Save embeddings with address ===
print("[4] Saving embeddings...")
df_emb = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
df_emb.insert(0, "address", df['address'])
df_emb.to_csv("/home/s3986160/master-thesis/Results/user_embeddings_minilm.csv", index=False)
print("✅ Saved to user_embeddings_minilm.csv")
