import os
import csv
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import selfies as sf  # pip install selfies

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
#DATA_DIR    = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))
#DATA_DIR = "/mnt/d/akarmark/data"
# ─── 2) Determine data directory relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data"))
INPUT_CSV   = os.path.join(DATA_DIR, "sample_smiles.csv")       # should have columns: CID, SMILES, [SELFIES]
OUTPUT_CSV  = os.path.join(DATA_DIR, "smiles_fingerprints.csv")
N_BITS      = 1024  # fingerprint length
RADIUS      = 2     # ECFP4

# ─── FINGERPRINT FUNCTION ──────────────────────────────────────────────────────
def mol_to_ecfp4(smiles_or_selfies: str, is_selfies: bool=False) -> np.ndarray:
    """
    Convert a SMILES (or SELFIES, if is_selfies=True) string to a 1024-bit ECFP4 fingerprint array.
    """
    if is_selfies:
        try:
            smiles = sf.decoder(smiles_or_selfies)
        except Exception as e:
            raise ValueError(f"Invalid SELFIES: {smiles_or_selfies}") from e
    else:
        smiles = smiles_or_selfies

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES after decoding: {smiles}")
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=RADIUS, nBits=N_BITS)
    arr = np.zeros((N_BITS,), dtype=np.float32)
    for idx, bit in enumerate(bitvect):
        arr[idx] = bit
    return arr

# ─── MAIN SCRIPT ─────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(INPUT_CSV, usecols=["CID", "SMILES"], dtype=str)
    # if you also have SELFIES column, do: usecols=["CID","SMILES","SELFIES"]
    n = len(df)
    print(f"Loaded {n} molecules from {INPUT_CSV}")

    # Prepare output CSV header
    header = ["CID"] + [f"fp_{i}" for i in range(N_BITS)]
    with open(OUTPUT_CSV, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)

        for idx, row in df.iterrows():
            cid    = row["CID"]
            smiles = row["SMILES"]
            # If you have SELFIES too and want to do both, you could loop twice:
            # selfies = row.get("SELFIES", "")
            try:
                fp_vec = mol_to_ecfp4(smiles, is_selfies=False)
            except ValueError as e:
                print(f"  ⚠️ Skipping {cid}: {e}")
                continue

            writer.writerow([cid] + fp_vec.tolist())
            if (idx+1) % 100 == 0 or idx == n-1:
                print(f"  • Processed {idx+1}/{n}")

    print(f"✅ Done. Wrote fingerprint embeddings to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
