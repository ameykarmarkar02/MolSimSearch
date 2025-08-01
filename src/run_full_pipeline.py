#!/usr/bin/env python3
import sys
import requests
import pandas as pd
import matplotlib.pyplot as plt
import selfies

# ─── 1) CONFIG ────────────────────────────────────────────────────────────────
API_URL       = "http://localhost:8000"
# Map internal endpoint paths to friendly labels for reporting and plotting
ENDPOINT_LABELS = {
    "semantic_search": "chemBERTa_smiles",
    "selfies_semantic_search": "chemBERTa_selfies",
    # "semantic_search_SELFormer": "SELFormer_smiles",
    "selfies_semantic_search_SELFormer": "SELFormer_selfies",
    # "/fingerprints_cosine_investigate": "fingerprints_cosine_investigate",
    "fingerprint_semantic_search": "fingerprints_cosine",
    "fingerprint_semantic_search_Tanimoto": "fingerprints_Tanimoto",
    # "fingerprints_Tanimoto_investigate": "fingerprints_Tanimoto",
}
ENDPOINTS     = list(ENDPOINT_LABELS.keys())
# k-values for your plots
K_VALUES      = [3,5,8,10,15,20,25,35]
BATCH_SIZE    = 5  # number of parallel threads
INPUT_FILE    = '/mnt/d/akarmark/Results/Data/queries.txt'
FULL_RESULTS  = '/mnt/d/akarmark/Results/Data/results_full.csv'
GED_SUMMARY   = '/mnt/d/akarmark/Results/Data/ged_summary.csv'
HAM_SUMMARY   = '/mnt/d/akarmark/Results/Data/hamming_summary.csv'
GED_PLOT_PNG  = '/mnt/d/akarmark/Results/Data/mean_GED_vs_k.png'
HAM_PLOT_PNG  = '/mnt/d/akarmark/Results/Data/mean_Hamming_vs_k.png'

# ─── 2) LOAD QUERIES ─────────────────────────────────────────────────────────
try:
    with open(INPUT_FILE) as f:
        queries = [l.strip() for l in f if l.strip()]
    if not queries:
        raise ValueError("No SMILES found in queries file")
except Exception as e:
    print(f"[ERROR] reading {INPUT_FILE}: {e}", file=sys.stderr)
    sys.exit(1)

# ─── 3) WORKER ────────────────────────────────────────────────────────────────
def process_smiles(smi):
    """Fetch top hits for all endpoints for one SMILES."""
    rows = []
    for ep in ENDPOINTS:
        url = f"{API_URL}/{ep}"
        if ep in ("selfies_semantic_search", "selfies_semantic_search_SELFormer"):
            query_texts = [selfies.encoder(smi)]
        else:
            query_texts = [smi]
        payload = {"texts": query_texts, "top_k": max(K_VALUES)}

        try:
            r = requests.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[ERROR] {ep} @ smi={smi}: {e}", file=sys.stderr)
            continue

        ids       = data.get("ids",       [[]])[0]
        if ep in ("selfies_semantic_search", "selfies_semantic_search_SELFormer"):
            hits = data.get("selfies", [[]])[0]
        else:
            hits = data.get("smiles",  [[]])[0]
        distances = data.get("distances", [[]])[0]
        geds      = data.get("ged",       [[None] * len(ids)])[0]
        hamms     = data.get("hamming",   [[None] * len(ids)])[0]
        raw_tan   = data.get("tanimoto")
        raw_cos   = data.get("cosine")
        raw_fp_q  = data.get("fingerprints_query", [[]])
        raw_fp_h  = data.get("fingerprints_hits", [[ ]])
        raw_fp_qf = data.get("fingerprints_query_floats", [[]])
        raw_fp_hf = data.get("fingerprints_hits_floats", [[ ]])

        tanms = raw_tan[0] if isinstance(raw_tan, list) and raw_tan else [None] * len(ids)
        cosims = raw_cos[0] if isinstance(raw_cos, list) and raw_cos else [None] * len(ids)

        fp_query = None
        fp_query_float = None
        if isinstance(raw_fp_q, list) and raw_fp_q and isinstance(raw_fp_q[0], list):
            fp_query = raw_fp_q[0]
        if isinstance(raw_fp_qf, list) and raw_fp_qf and isinstance(raw_fp_qf[0], list):
            fp_query_float = raw_fp_qf[0]

        label = ENDPOINT_LABELS[ep]

        for rank, mid in enumerate(ids, start=1):
            fp_hit = None
            fp_hit_float = None
            if (isinstance(raw_fp_h, list) and raw_fp_h and isinstance(raw_fp_h[0], list)
                and len(raw_fp_h[0]) >= rank):
                fp_hit = raw_fp_h[0][rank-1]
            if (isinstance(raw_fp_hf, list) and raw_fp_hf and isinstance(raw_fp_hf[0], list)
                and len(raw_fp_hf[0]) >= rank):
                fp_hit_float = raw_fp_hf[0][rank-1]

            rows.append({
                "query_smiles": smi,
                "endpoint":     label,
                "rank":         rank,
                "hit_id":       mid,
                "hit_smiles":   hits[rank-1]      if rank-1 < len(hits)      else None,
                "distance":     distances[rank-1] if rank-1 < len(distances) else None,
                "GED":          geds[rank-1]      if rank-1 < len(geds)      else None,
                "Hamming":      hamms[rank-1]     if rank-1 < len(hamms)     else None,
                "Tanimoto":     tanms[rank-1]     if rank-1 < len(tanms)     else None,
                "Cosine":       cosims[rank-1]    if rank-1 < len(cosims)    else None,
                "query_fp":     fp_query,
                "hit_fp":       fp_hit,
                "query_fp_float": fp_query_float,
                "hit_fp_float":   fp_hit_float,
            })
    return rows

# ─── 4) FETCH ALL RESULTS (parallel) ─────────────────────────────────────────
from concurrent.futures import ThreadPoolExecutor, as_completed
all_records = []
with ThreadPoolExecutor(max_workers=BATCH_SIZE) as exe:
    futures = {exe.submit(process_smiles, smi): smi for smi in queries}
    for fut in as_completed(futures):
        smi = futures[fut]
        try:
            all_records.extend(fut.result())
        except Exception as e:
            print(f"[ERROR] processing {smi}: {e}", file=sys.stderr)

# ─── 5) SAVE FULL RESULTS ────────────────────────────────────────────────────
df_full = pd.DataFrame.from_records(
    all_records,
    columns=[
        "query_smiles", "endpoint", "rank",
        "hit_id", "hit_smiles", "distance",
        "GED", "Hamming", "Tanimoto", "Cosine",
        "query_fp", "hit_fp",
        "query_fp_float", "hit_fp_float"
    ]
)
df_full.to_csv(FULL_RESULTS, index=False)
print(f"DONE: wrote full results → {FULL_RESULTS} ({len(df_full)} rows)")

# ─── 5.1) FILTER INVALID HITS ───────────────────────────────────────────────
df_valid = df_full.loc[(df_full["Hamming"] != -1) & (df_full["GED"].notna())].copy()
print(f"→ Dropped invalid hits: {len(df_full) - len(df_valid)} rows removed")

# ─── 6) SUMMARY STATS BY SUBSETS OF RANK ────────────────────────────────────
ged_rows = []
ham_rows = []
for k in K_VALUES:
    subset = df_valid[df_valid["rank"] <= k]
    grp = subset.groupby("endpoint")
    for ep, sub in grp:
        sged = sub["GED"].astype(float)
        sham = sub["Hamming"].astype(float)
        ged_rows.append({
            "top_k":       k,
            "endpoint":    ep,
            "mean_GED":    sged.mean(),
            "median_GED":  sged.median(),
            "std_GED":     sged.std(ddof=0),
            "count_GED":   len(sged),
        })
        ham_rows.append({
            "top_k":         k,
            "endpoint":      ep,
            "mean_Hamming":  sham.mean(),
            "median_Hamming":sham.median(),
            "std_Hamming":   sham.std(ddof=0),
            "count_Hamming": len(sham),
        })

# Save summaries
df_ged_sum = pd.DataFrame(ged_rows)
df_ham_sum = pd.DataFrame(ham_rows)
df_ged_sum.to_csv(GED_SUMMARY, index=False)
df_ham_sum.to_csv(HAM_SUMMARY, index=False)
print(f"DONE: wrote GED summary → {GED_SUMMARY}")
print(f"DONE: wrote Hamming summary → {HAM_SUMMARY}")

# ─── 7) PLOT: Mean GED vs. top_k ─────────────────────────────────────────────
pivot_ged = df_ged_sum.pivot(index="top_k", columns="endpoint", values="mean_GED")
plt.figure(figsize=(8,5))
for ep in pivot_ged.columns:
    plt.plot(pivot_ged.index, pivot_ged[ep], marker="o", label=ep)
plt.xlabel("Top-k")
plt.ylabel("Mean GED")
plt.title("Mean GED vs. Top-k Hits")
plt.xticks(K_VALUES)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(GED_PLOT_PNG, dpi=300)
plt.close()
print(f"DONE: saved GED plot → {GED_PLOT_PNG}")

# ─── 8) PLOT: Mean Hamming vs. top_k ───────────────────────────────────────
pivot_ham = df_ham_sum.pivot(index="top_k", columns="endpoint", values="mean_Hamming")
plt.figure(figsize=(8,5))
for ep in pivot_ham.columns:
    plt.plot(pivot_ham.index, pivot_ham[ep], marker="o", label=ep)
plt.xlabel("Top-k")
plt.ylabel("Mean Hamming")
plt.title("Mean Hamming vs. Top-k Hits")
plt.xticks(K_VALUES)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(HAM_PLOT_PNG, dpi=300)
plt.close()
print(f"DONE: saved Hamming plot → {HAM_PLOT_PNG}")
