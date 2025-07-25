import requests
import pandas as pd
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

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
TOP_K = 5
BATCH_SIZE = 1
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

def process_smiles(smi):
    """Query all endpoints for one SMILES and return a list of hit-record dicts."""
    recs = []
    for endpoint in ENDPOINTS:
        url = f"{API_URL}/{endpoint}"
        payload = {"texts": [smi], "top_k": TOP_K}
        try:
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[ERROR] {endpoint} failed for {smi}: {e}", file=sys.stderr)
            continue

        ids       = data.get("ids",       [[]])[0]
        hits      = data.get("smiles",    [[]])[0]
        distances = data.get("distances", [[]])[0]
        geds      = data.get("ged",       [[None] * len(ids)])[0]
        hamms     = data.get("hamming",   [[None] * len(ids)])[0]

        for rank, hit_id in enumerate(ids, start=1):
            recs.append({
                "query_smiles": smi,
                "endpoint": endpoint,
                "rank": rank,
                "hit_id": hit_id,
                "hit_smiles": hits[rank-1] if rank-1 < len(hits) else None,
                "distance": distances[rank-1] if rank-1 < len(distances) else None,
                "GED":      geds[rank-1]      if rank-1 < len(geds)      else None,
                "Hamming":  hamms[rank-1]     if rank-1 < len(hamms)     else None,
            })
    return recs

# 3) Process in batches of 5, with 5 threads each batch
all_records = []
for i in range(0, len(queries), BATCH_SIZE):
    batch = queries[i:i+BATCH_SIZE]
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as exe:
        futures = {exe.submit(process_smiles, smi): smi for smi in batch}
        for fut in as_completed(futures):
            smi = futures[fut]
            try:
                all_records.extend(fut.result())
            except Exception as e:
                print(f"[ERROR] batch item {smi} raised {e}", file=sys.stderr)

# 4) Save to CSV (exactly as original)
df = pd.DataFrame.from_records(
    all_records,
    columns=[
        "query_smiles", "endpoint", "rank",
        "hit_id", "hit_smiles",
        "distance", "GED", "Hamming"
    ]
)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Done — wrote {len(df)} rows to {OUTPUT_FILE}")
