import os
import pickle
import pandas as pd
import numpy as np
import faiss

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(os.path.dirname(__file__), "..", "data")
SMILES_CSV      = os.path.join(DATA_DIR, "sample_embeddings.csv")
SELFIES_CSV     = os.path.join(DATA_DIR, "selfies_embeddings.csv")
OUT_DIR         = os.path.join(DATA_DIR, "faiss_indexes")
DIM             = 768
INDEX_TYPE      = faiss.IndexFlatIP  # inner-product on L2-normalized vectors

os.makedirs(OUT_DIR, exist_ok=True)

def load_and_normalize(path):
    df = pd.read_csv(path)
    ids = df["CID"].astype(str).tolist()
    vecs = df.drop(columns=["CID"]).to_numpy(dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.clip(norms, 1e-8, None)
    return ids, vecs

def build_and_save(ids, vecs, prefix):
    # build
    idx = INDEX_TYPE(DIM)
    idx.add(vecs)
    # save index
    idx_path = os.path.join(OUT_DIR, f"{prefix}.index")
    faiss.write_index(idx, idx_path)
    # save id mapping
    with open(os.path.join(OUT_DIR, f"{prefix}_ids.pkl"), "wb") as f:
        pickle.dump(ids, f)
    print(f"Built & saved {prefix} index ({len(ids)} vectors) → {idx_path}")

if __name__ == "__main__":
    # SMILES
    ids_sm, vecs_sm = load_and_normalize(SMILES_CSV)
    build_and_save(ids_sm, vecs_sm, "smiles")

    # SELFIES
    ids_sf, vecs_sf = load_and_normalize(SELFIES_CSV)
    build_and_save(ids_sf, vecs_sf, "selfies")
