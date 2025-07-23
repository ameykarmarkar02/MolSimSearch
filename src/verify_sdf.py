from rdkit import Chem
import os

# Path to the decompressed SDF — use the exact filename you downloaded
sdf_path = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "Compound_000000001_000500000.sdf")
)

# Sanity‐check RDKit can read the first few molecules
suppl = Chem.SDMolSupplier(sdf_path)
n_total = 0
n_valid = 0

for mol in suppl:
    n_total += 1
    if mol is not None:
        n_valid += 1
    if n_total >= 10:  # just read first 10 for speed
        break

print(f"Read {n_total} records, {n_valid} valid molecules (first {n_total} only).")
