import os
import csv
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
#DATA_DIR   = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))
DATA_DIR = "/mnt/d/akarmark/data"
INPUT_CSV  = os.path.join(DATA_DIR, "sample_smiles.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "sample_embeddings.csv")
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
BATCH_SIZE = 64

# ─── DEVICE SETUP ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── LOAD MODEL & TOKENIZER ───────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# ─── EMBEDDING FUNCTION ───────────────────────────────────────────────────────
def embed_smiles_batch(smiles_list):
    # Tokenize and move to device
    inputs = tokenizer(smiles_list,
                       return_tensors="pt",
                       padding=True,
                       truncation=True,
                       max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean‐pool over sequence dimension
        embs = outputs.last_hidden_state.mean(dim=1)
        return embs.cpu().numpy()

# ─── MAIN SCRIPT ─────────────────────────────────────────────────────────────
def main():
    # Load input CSV
    df = pd.read_csv(INPUT_CSV, usecols=["CID", "SMILES"])
    n = len(df)
    print(f"Loaded {n} molecules from {INPUT_CSV}")

    # Prepare output CSV
    header = ["CID"] + [f"emb_{i}" for i in range(model.config.hidden_size)]
    with open(OUTPUT_CSV, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)

        # Process in batches
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            batch = df.iloc[start:end]
            smiles_list = batch["SMILES"].tolist()
            cids        = batch["CID"].tolist()

            # Generate embeddings
            embeddings = embed_smiles_batch(smiles_list)

            # Write rows: [CID, emb_0, emb_1, ..., emb_767]
            for cid, emb in zip(cids, embeddings):
                writer.writerow([cid] + emb.tolist())

            print(f"  • Processed molecules {start+1}–{end}")

    print(f"✅ Done. Wrote embeddings to {OUTPUT_CSV}")

    # Quick verify
    out_df = pd.read_csv(OUTPUT_CSV, nrows=5)
    print("Sample output shape:", out_df.shape)
    print(out_df.iloc[0, 1:6].values, "...")  # show first 5 dims

if __name__ == "__main__":
    main()
