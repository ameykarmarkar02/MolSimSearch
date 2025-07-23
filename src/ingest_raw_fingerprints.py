import os
import pickle
import pandas as pd
import numpy as np
import faiss

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR    = "/mnt/d/akarmark/data"
SMILES_CSV  = os.path.join(DATA_DIR, "smiles_fingerprints.csv")
SELFIES_CSV = os.path.join(DATA_DIR, "selfies_fingerprints.csv")
OUT_DIR     = os.path.join(DATA_DIR, "faiss_indexes")
# Use Inner Product on L2-normalized vectors
INDEX_TYPE  = faiss.IndexFlatIP
# Number of rows to stream at once to limit memory use
CHUNK_SIZE  = 100_000

os.makedirs(OUT_DIR, exist_ok=True)


def stream_and_build(csv_path: str, prefix: str):
    """
    Stream CSV in chunks, normalize vectors, ensure contiguity, and incrementally build FAISS index.
    """
    all_ids = []
    idx = None

    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):
        # Extract IDs and vectors
        ids = chunk["CID"].astype(str).tolist()
        vecs = chunk.drop(columns=["CID"]).to_numpy(dtype=np.float32)

        # L2-normalize each row
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.clip(norms, 1e-8, None)

        # Ensure C-contiguous float32 array for FAISS
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)

        # Initialize index on first chunk
        if idx is None:
            dim = vecs.shape[1]
            print(f"Initializing FAISS index '{prefix}' with dimension = {dim}")
            idx = INDEX_TYPE(dim)

        # Add vectors to index
        print(f"Adding {vecs.shape[0]} vectors to '{prefix}'")
        idx.add(vecs)

        # Collect IDs
        all_ids.extend(ids)

    # Save the built index
    idx_path = os.path.join(OUT_DIR, f"{prefix}.index")
    faiss.write_index(idx, idx_path)
    print(f"Saved FAISS index to {idx_path}")

    # Save ID mapping via pickle
    ids_path = os.path.join(OUT_DIR, f"{prefix}_ids.pkl")
    with open(ids_path, "wb") as f:
        pickle.dump(all_ids, f)
    print(f"Saved IDs to {ids_path}  (total {len(all_ids)} entries)")


if __name__ == "__main__":
    # SMILES fingerprints
    stream_and_build(SMILES_CSV, "smiles_fingerprints_raw")

    # SELFIES fingerprints
    stream_and_build(SELFIES_CSV, "selfies_fingerprints_raw")
