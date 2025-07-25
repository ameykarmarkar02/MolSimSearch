import requests
import pandas as pd
import sys

# 1) Configuration
API_URL = "http://localhost:8000"
ENDPOINTS = [
    "semantic_search",
    "semantic_search_SELFormer",
    "selfies_semantic_search",
    # "selfies_semantic_search_SELFormer",
    "fingerprint_semantic_search",
    "fingerprint_semantic_search_Tanimoto",
]
TOP_K = 35
INPUT_FILE = '/mnt/d/akarmark/Results/Data/queries.txt'
OUTPUT_FILE = '/mnt/d/akarmark/Results/Data/results.csv'

# 2) Read queries
try:
    with open(INPUT_FILE, "r") as f:
        queries = [line.strip() for line in f if line.strip()]
    if not queries:
        raise ValueError("No SMILES found in queries.txt")
except Exception as e:
    print(f"Error reading {INPUT_FILE}: {e}", file=sys.stderr)
    sys.exit(1)

# 3) Collect results
records = []

for smi in queries:
    for endpoint in ENDPOINTS:
        url = f"{API_URL}/{endpoint}"
        payload = {"texts": [smi], "top_k": TOP_K}
        try:
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Request to {endpoint} failed for {smi}: {e}", file=sys.stderr)
            continue

        # unpack
        ids       = data.get("ids",       [[]])[0]
        hits      = data.get("smiles",    [[]])[0]
        distances = data.get("distances", [[]])[0]
        geds      = data.get("ged",       [[None] * len(ids)])[0]
        hamms     = data.get("hamming",   [[None] * len(ids)])[0]

        # if fingerprint endpoints return a "similarity" or "sims" field:
        # sims = data.get("similarity", data.get("sims", [[None] * len(ids)]))[0]

        # build one record per hit
        for rank, hit_id in enumerate(ids, start=1):
            records.append({
                "query_smiles": smi,
                "endpoint": endpoint,
                "rank": rank,
                "hit_id": hit_id,
                "hit_smiles": hits[rank-1] if rank-1 < len(hits) else None,
                "distance": distances[rank-1] if rank-1 < len(distances) else None,
                "GED":      geds[rank-1]      if rank-1 < len(geds)      else None,
                "Hamming":  hamms[rank-1]     if rank-1 < len(hamms)     else None,
                # "similarity": sims[rank-1]    if sims is not None and rank-1 < len(sims) else None,
            })

# 4) Save to CSV
df = pd.DataFrame.from_records(records,
    columns=[
        "query_smiles", "endpoint", "rank",
        "hit_id", "hit_smiles",
        "distance", "GED", "Hamming"
    ]
)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Done — wrote {len(df)} rows to {OUTPUT_FILE}")
