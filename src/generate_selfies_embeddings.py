# File: src/generate_selfies_embeddings.py

import os
import csv
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Paths
BASE_DIR    = os.path.dirname(__file__)
DATA_DIR    = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))
INPUT_CSV   = os.path.join(DATA_DIR, "sample_converted.csv")
OUTPUT_CSV  = os.path.join(DATA_DIR, "selfies_embeddings.csv")

# Model
MODEL_NAME  = "seyonec/ChemBERTa-zinc-base-v1"
BATCH_SIZE  = 64

# ─── DEVICE SETUP ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── LOAD MODEL & TOKENIZER ───────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# ─── EMBEDDING FUNCTION ───────────────────────────────────────────────────────
def embed_selfies_batch(selfies_list):
    inputs = tokenizer(
        selfies_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embs = outputs.last_hidden_state.mean(dim=1)  # (batch, hidden_dim)
    return embs.cpu().numpy()

# ─── MAIN SCRIPT ─────────────────────────────────────────────────────────────
def main():
    # Verify input
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    # Load the combined CSV with SELFIES
    df = pd.read_csv(INPUT_CSV, usecols=["CID", "SELFIES"])
    n = len(df)
    print(f"Loaded {n} molecules (CID + SELFIES) from {INPUT_CSV}")

    # Prepare output CSV header
    hidden_size = model.config.hidden_size
    header = ["CID"] + [f"emb_{i}" for i in range(hidden_size)]

    # Open output file
    with open(OUTPUT_CSV, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)

        # Process in batches
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            batch = df.iloc[start:end]
            cids        = batch["CID"].tolist()
            selfies_lst = batch["SELFIES"].tolist()

            # Generate embeddings
            embeddings = embed_selfies_batch(selfies_lst)

            # Write rows
            for cid, emb in zip(cids, embeddings):
                writer.writerow([cid] + emb.tolist())

            print(f"  • Processed SELFIES {start+1}–{end}")

    print(f"✅ Done. Wrote SELFIES embeddings to {OUTPUT_CSV}")

    # Quick verify
    out_df = pd.read_csv(OUTPUT_CSV, nrows=5)
    print("Sample output shape:", out_df.shape)
    print("First embedding dims:", out_df.iloc[0, 1:6].values)

if __name__ == "__main__":
    main()
