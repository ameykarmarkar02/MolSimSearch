# app.py

from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.utils.load_faiss import load_faiss_index
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss
import pickle
import os
import pandas as pd
from src.queries.parser import parse_complex_query
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.DataStructs import ExplicitBitVect
from src.utils.load_faiss import load_faiss_index
import selfies as sf
import time
import networkx as nx
from threading import Thread, Event, Lock



app = FastAPI(
    title="Molecule Semantic Search API",
    version="0.1.0",
    description="Embed texts and search FAISS index for molecular similarity."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────── Load Model and Indexes ───────────────

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()


SMILES_INDEX_PATH  = "/mnt/d/akarmark/data/faiss_indexes/smiles.index"
SMILES_IDMAP_PATH  = "/mnt/d/akarmark/data/faiss_indexes/smiles_ids.pkl"
smiles_index, smiles_id_map = load_faiss_index(SMILES_INDEX_PATH, SMILES_IDMAP_PATH)

SELFIES_INDEX_PATH = "/mnt/d/akarmark/data/faiss_indexes/selfies.index"
SELFIES_IDMAP_PATH = "/mnt/d/akarmark/data/faiss_indexes/selfies_ids.pkl"
selfies_index, selfies_id_map = load_faiss_index(SELFIES_INDEX_PATH, SELFIES_IDMAP_PATH)

# ─────────────── SELFormer Model & Index ───────────────
SELF_MODEL = "HUBioDataLab/SELFormer"

# SELFormer tokenizer and model (use safetensors)
sel_tokenizer = AutoTokenizer.from_pretrained(SELF_MODEL)
sel_model     = AutoModel.from_pretrained(SELF_MODEL, use_safetensors=True).to(DEVICE)
sel_model.eval()



# # New SELFormer FAISS index
# SMILES_SELF_INDEX_PATH    = "/mnt/d/akarmark/data/faiss_indexes/smiles_SELFormer.index"
# SMILES_SELF_IDMAP_PATH    = "/mnt/d/akarmark/data/faiss_indexes/smiles_SELFormer_ids.pkl"
# smiles_SELFormer_index, smiles_SELFormer_id_map = load_faiss_index(
#     SMILES_SELF_INDEX_PATH,
#     SMILES_SELF_IDMAP_PATH
# )

#New SELFormer SELFIES index
SELFIES_SELF_INDEX_PATH = "/mnt/d/akarmark/data/faiss_indexes/selfies_SELFormer.index"
SELFIES_SELF_IDMAP_PATH = "/mnt/d/akarmark/data/faiss_indexes/selfies_SELFormer_ids.pkl"
selfies_SELFormer_index, selfies_SELFormer_id_map = load_faiss_index(
    SELFIES_SELF_INDEX_PATH,
    SELFIES_SELF_IDMAP_PATH
)


CONVERTED_CSV = "/mnt/d/akarmark/data/sample_converted.csv"
df_mols = pd.read_csv(CONVERTED_CSV)

smiles_to_internal = { df_mols.iloc[idx]["SMILES"]: idx for idx in range(len(df_mols)) }

selfies_df = pd.read_csv(CONVERTED_CSV)
selfies_map = dict(zip(selfies_df["CID"].astype(str), selfies_df["SELFIES"]))

# ─────────────── Load Precomputed Fingerprints ───────────────

print("🔄 Starting to load precomputed SMILES fingerprints...")

# FINGERPRINTS_CSV = "/mnt/d/akarmark/data/smiles_fingerprints.csv"

# print("🔄 Reading CSV only...")
# start = time.time()

# fp_df = pd.read_csv(FINGERPRINTS_CSV)
# # chunks = pd.read_csv(FINGERPRINTS_CSV, chunksize=5000)
# # fp_df = pd.concat(chunks, ignore_index=True)

# print(f"✅ Done reading. Rows: {len(fp_df)} | Time: {time.time() - start:.2f}s")

# #fp_df = pd.read_csv(FINGERPRINTS_CSV)
# # chunks = pd.read_csv(FINGERPRINTS_CSV, chunksize=5000)
# # fp_df = pd.concat(chunks, ignore_index=True)


# print(f"✅ Loaded fingerprint CSV with {len(fp_df)} rows.")

# all_fps = []
# for i in range(len(fp_df)):
#     bits = fp_df.iloc[i].values.astype(bool)
#     fp = ExplicitBitVect(len(bits))
#     for j, bit in enumerate(bits):
#         if bit:
#             fp.SetBit(j)
#     all_fps.append(fp)
    
#     if (i + 1) % 1 == 0:
#         print(f"   → Loaded {i + 1} fingerprints...")

# print(f"✅ Finished loading all {len(all_fps)} SMILES fingerprints.")

FINGERPRINTS_CSV = "/mnt/d/akarmark/data/smiles_fingerprints.csv"

print("🔄 Starting to load precomputed SMILES fingerprints...")

start = time.time()

all_fps = []
chunk_size = 5000

try:
    for chunk_idx, chunk in enumerate(pd.read_csv(FINGERPRINTS_CSV, chunksize=chunk_size)):
        print(f"📦 Reading chunk {chunk_idx + 1}...")

        for i, row in enumerate(chunk.itertuples(index=False)):
            bits = [bool(b) for b in row]
            fp = ExplicitBitVect(len(bits))
            for j, bit in enumerate(bits):
                if bit:
                    fp.SetBit(j)
            all_fps.append(fp)

            # Print every 100 molecules (you can change this interval)
            if (len(all_fps)) % 100 == 0:
                print(f"   → Loaded {len(all_fps)} fingerprints so far...")

except Exception as e:
    print(f"❌ Error while processing: {e}")

end = time.time()

print(f"✅ Finished loading all {len(all_fps)} SMILES fingerprints.")
print(f"⏱ Total time: {end - start:.2f} seconds")
# ─────────── Build SMILES→Fingerprint Lookup ───────────
smiles_to_fp = dict(zip(df_mols["SMILES"].tolist(), all_fps))

# all_selfies_fps = []
# for sf_str in selfies_df["SELFIES"]:
#     try:
#         smi = sf.decoder(sf_str)
#         mol = Chem.MolFromSmiles(smi)
#     except:
#         mol = None
#     fp = ExplicitBitVect(1024) if mol is None else AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
#     all_selfies_fps.append(fp)

# Load fingerprint index and ID map once at startup
fingerprint_index, fingerprint_id_map = load_faiss_index(
    "/mnt/d/akarmark/data/faiss_indexes/smiles_fingerprints_raw.index",
    "/mnt/d/akarmark/data/faiss_indexes/smiles_fingerprints_raw_ids.pkl"
)
# Load fingerprint index and ID map once at startup
selfies_fp_index, selfies_fp_id_map = load_faiss_index(
    "/mnt/d/akarmark/data/faiss_indexes/selfies_fingerprints_raw.index",
    "/mnt/d/akarmark/data/faiss_indexes/selfies_fingerprints_raw_ids.pkl"
)

# ─────────────── Request/Response Schemas ───────────────

class EmbedRequest(BaseModel):
    texts: list[str]

class SearchRequest(BaseModel):
    vectors: list[list[float]]
    top_k: int = 10

class SemanticSearchRequest(BaseModel):
    texts: list[str]
    top_k: int = 10

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]

class SearchResponse(BaseModel):
    ids: List[List[str]]
    distances: List[List[float]]
    smiles: List[List[str]] = []
    queries: List[str] = []
    ged: Optional[List[List[Optional[float]]]] = None
    hamming: Optional[List[List[int]]] = None
    tanimoto: Optional[List[List[float]]] = None
    cosine: Optional[List[List[float]]] = None  # Cosine similarity scores
    fingerprints_query: Optional[List[Optional[List[int]]]] = None     # ECFP4 bits for each query
    fingerprints_hits: Optional[List[List[Optional[List[int]]]]] = None  # ECFP4 bits for each hit
    fingerprints_query_floats: Optional[List[List[float]]] = None      # FAISS-normalized float vectors for each query
    fingerprints_hits_floats: Optional[List[List[List[float]]]] = None  # FAISS-normalized float vectors for each hit



class SelfiesSearchResponse(BaseModel):
    ids: list[list[str]]
    distances: list[list[float]]
    selfies: list[list[str]]
    queries: list[str] = []
    #ged: Optional[List[List[float]]] = None  # ← make optional
    ged: Optional[list[list[Optional[float]]]] = None  # allow None values inside GED matrix
    hamming: Optional[List[List[int]]] = None          # Hamming distances


class AddToIndexRequest(BaseModel):
    texts: list[str]
    ids: list[str]

class ComplexQueryRequest(BaseModel):
    query: str

# ─────────────── Utility Functions ───────────────

# def load_selfies_SELFormer_index_async():
#     global selfies_SELFormer_index, selfies_SELFormer_id_map
#     print("🔄 Lazy-loading SELFIES SELFormer index in background...")
#     selfies_SELFormer_index, selfies_SELFormer_id_map = load_faiss_index(
#         "/mnt/d/akarmark/data/faiss_indexes/selfies_SELFormer.index",
#         "/mnt/d/akarmark/data/faiss_indexes/selfies_SELFormer_ids.pkl"
#     )
#     selfies_self_loaded.set()
#     print("✅ SELFIES SELFormer index loaded.")

def _embed_texts(texts: list[str]) -> np.ndarray:
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return pooled.cpu().numpy().astype("float32")

def _embed_text(text: str) -> np.ndarray:
    return _embed_texts([text])[0]

def _search_index(idx: faiss.Index, id_map: dict[int, str], vectors: np.ndarray, top_k: int):
    D, I = idx.search(vectors, top_k)
    ids = [[id_map.get(i, str(i)) for i in row] for row in I.tolist()]
    return ids, D.tolist()


def get_selfies_for_ids(ids_list: list[list[str]]) -> list[list[str]]:
    return [[selfies_map.get(str(cid), "") for cid in batch] for batch in ids_list]

from rdkit import DataStructs

def generate_fingerprint_vector(smiles: str) -> np.ndarray:
    """
    Generate a normalized 1024-bit Morgan fingerprint vector from a SMILES string.
    Returns numpy float32 vector or raises ValueError on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    arr = np.zeros((1024,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr /= norm
    return arr


# def mol_to_nx(smiles: str) -> nx.Graph:
#     """Convert a SMILES string to a NetworkX graph (unit‑cost GED)."""
#     mol = Chem.MolFromSmiles(smiles)
#     G = nx.Graph()
#     for atom in mol.GetAtoms():
#         G.add_node(atom.GetIdx(), label=atom.GetSymbol())
#     for bond in mol.GetBonds():
#         G.add_edge(
#             bond.GetBeginAtomIdx(),
#             bond.GetEndAtomIdx(),
#             label=str(bond.GetBondType())
#         )
#     return G

def mol_to_nx(smiles: str) -> nx.Graph:
    """Convert a SMILES string to a NetworkX graph (unit-cost GED)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # signal that this SMILES can’t be parsed
        raise ValueError(f"Invalid SMILES: {smiles}")

    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), label=atom.GetSymbol())
    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            label=str(bond.GetBondType())
        )
    return G

# Helper to compute Hamming distance between two RDKit fingerprints
def hamming_distance(fp1, fp2) -> int:
    # Assumes fp1 and fp2 are ExplicitBitVect of same length
    return sum(fp1.GetBit(i) != fp2.GetBit(i) for i in range(fp1.GetNumBits()))


# ─────────────── Health Check ───────────────

@app.get("/health")
def health_check():
    return {"status": "ok"}

# ─────────────── SMILES Endpoints ───────────────

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    embeddings = _embed_texts(req.texts)
    return {"embeddings": embeddings.tolist()}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    vectors = np.array(req.vectors, dtype="float32")
    ids, dists = _search_index(smiles_index, smiles_id_map, vectors, req.top_k)

    smiles = []
    for row in ids:
        row_sm = []
        for mol_id in row:
            try:
                idx = int(mol_id)
                row_sm.append(df_mols.iloc[idx]["SMILES"])
            except:
                row_sm.append("UNKNOWN")
        smiles.append(row_sm)

    return {"ids": ids, "distances": dists, "smiles": smiles, "queries": []}

@app.post("/semantic_search", response_model=SearchResponse)
def semantic_search(req: SemanticSearchRequest):
    # 1) Build or retrieve normalized query embeddings
    query_vecs = []
    for smi in req.texts:
        row = df_mols[df_mols["SMILES"] == smi]
        stored_vec = None
        if not row.empty:
            try:
                embedding_cols = [c for c in df_mols.columns if c.startswith("emb_")]
                vec_values = row[embedding_cols].values.astype("float32").flatten()
                if vec_values.size == 768 and not np.isnan(vec_values).any():
                    stored_vec = vec_values / np.linalg.norm(vec_values)
            except Exception:
                pass
        fresh_vec = _embed_texts([smi])[0].astype("float32")
        fresh_vec /= np.linalg.norm(fresh_vec)
        query_vecs.append(stored_vec if stored_vec is not None else fresh_vec)

    embeddings = np.vstack(query_vecs)
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # 2) Run approximate nearest‑neighbor search (fetch top_k + possible extra to allow filtering)
    ids, dists = _search_index(smiles_index, smiles_id_map, normalized_embeddings, req.top_k + 1)

    # 3) Map back to SMILES strings
    smiles = []
    for row in ids:
        row_sm = []
        for mol_id in row:
            try:
                idx = int(mol_id)
                row_sm.append(df_mols.iloc[idx]["SMILES"])
            except:
                row_sm.append("UNKNOWN")
        smiles.append(row_sm)

    # 3.5) Filter out any hit identical to the query SMILES and limit to top_k
    filtered_ids = []
    filtered_dists = []
    filtered_smiles = []
    for q_smi, id_row, dist_row, smi_row in zip(req.texts, ids, dists, smiles):
        new_ids, new_dists_row, new_smiles_row = [], [], []
        for mol_id, dist, sm in zip(id_row, dist_row, smi_row):
            if sm != q_smi:
                new_ids.append(mol_id)
                new_dists_row.append(dist)
                new_smiles_row.append(sm)
            if len(new_ids) == req.top_k:
                break
        # If fewer than top_k, truncate or pad with UNKNOWN as needed
        filtered_ids.append(new_ids)
        filtered_dists.append(new_dists_row)
        filtered_smiles.append(new_smiles_row)

    # Replace original lists
    ids = filtered_ids
    dists = filtered_dists
    smiles = filtered_smiles

    # 4) Compute GED matrix with timeout
    ged_matrix: List[List[Optional[float]]] = []
    for q_smi, hits in zip(req.texts, smiles):
        Gq = mol_to_nx(q_smi)
        row_ged: List[Optional[float]] = []
        for hit_smi in hits:
            if hit_smi == "UNKNOWN":
                row_ged.append(None)
            else:
                Gr = mol_to_nx(hit_smi)
                ged_val = nx.graph_edit_distance(
                    Gq, Gr,
                    node_ins_cost=lambda _: 1,
                    node_del_cost=lambda _: 1,
                    node_subst_cost=lambda a, b: 0 if a['label'] == b['label'] else 1,
                    edge_ins_cost=lambda _: 1,
                    edge_del_cost=lambda _: 1,
                    timeout=300.0
                )
                row_ged.append(ged_val)
        ged_matrix.append(row_ged)

    # 5) Compute Hamming distances
    hamming_matrix: List[List[int]] = []
    for q_smi, hits in zip(req.texts, smiles):
        mol_q = Chem.MolFromSmiles(q_smi)
        fp_q = AllChem.GetMorganFingerprintAsBitVect(mol_q, 2, nBits=1024)
        row_ham: List[int] = []
        for hit_smi in hits:
            if hit_smi == "UNKNOWN":
                row_ham.append(-1)
            else:
                mol_h = Chem.MolFromSmiles(hit_smi)
                fp_h = AllChem.GetMorganFingerprintAsBitVect(mol_h, 2, nBits=1024)
                ham = hamming_distance(fp_q, fp_h)
                row_ham.append(ham)
        hamming_matrix.append(row_ham)

    # 6) Return everything
    return {
        "ids": ids,
        "distances": dists,
        "smiles": smiles,
        "queries": req.texts,
        "ged": ged_matrix,
        "hamming": hamming_matrix,
    }

# ─────────────── SELFormer Endpoint ───────────────
# @app.post("/semantic_search_SELFormer", response_model=SearchResponse)
# def semantic_search_SELFormer(req: SemanticSearchRequest):
#     # 1) Build or retrieve normalized query embeddings
#     query_vecs: List[np.ndarray] = []
#     for smi in req.texts:
#         stored_vec = None
#         row = df_mols[df_mols["SMILES"] == smi]
#         if not row.empty:
#             try:
#                 emb_cols = [c for c in df_mols.columns if c.startswith("emb_") and "SELFormer" in c]
#                 vec = row[emb_cols].values.astype("float32").flatten()
#                 if vec.size == sel_model.config.hidden_size and not np.isnan(vec).any():
#                     stored_vec = vec / np.linalg.norm(vec)
#             except:
#                 pass

#         inputs = sel_tokenizer(
#             [smi],
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=128
#         ).to(DEVICE)
#         with torch.no_grad():
#             outputs = sel_model(**inputs)
#             fresh = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0].astype("float32")
#         fresh /= np.linalg.norm(fresh)

#         query_vecs.append(stored_vec if stored_vec is not None else fresh)

#     emb_matrix = np.vstack(query_vecs)
#     norm_embs = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)

#     # 2) Run FAISS search (fetch top_k + 1 to allow filtering out self-hits)
#     ids, dists = _search_index(
#         smiles_SELFormer_index,
#         smiles_SELFormer_id_map,
#         norm_embs,
#         req.top_k + 1
#     )

#     # 3) Map back to SMILES
#     smiles_hits: List[List[str]] = []
#     for row_ids in ids:
#         row_hits: List[str] = []
#         for mid in row_ids:
#             try:
#                 row_hits.append(df_mols.iloc[int(mid)]["SMILES"])
#             except:
#                 row_hits.append("UNKNOWN")
#         smiles_hits.append(row_hits)

#     # 3.5) FILTER OUT QUERY MOLECULE FROM HITS AND TRUNCATE TO top_k
#     filtered_ids = []
#     filtered_dists = []
#     filtered_smiles_hits = []
#     for q_smi, id_row, dist_row, smi_row in zip(req.texts, ids, dists, smiles_hits):
#         new_ids, new_dists_row, new_smiles_row = [], [], []
#         for mol_id, dist, sm in zip(id_row, dist_row, smi_row):
#             if sm != q_smi:
#                 new_ids.append(mol_id)
#                 new_dists_row.append(dist)
#                 new_smiles_row.append(sm)
#             if len(new_ids) == req.top_k:
#                 break
#         filtered_ids.append(new_ids)
#         filtered_dists.append(new_dists_row)
#         filtered_smiles_hits.append(new_smiles_row)

#     # overwrite originals so steps 4–6 work unchanged
#     ids = filtered_ids
#     dists = filtered_dists
#     smiles_hits = filtered_smiles_hits

#     # 4) Compute GED matrix, skipping invalid SMILES
#     ged_matrix: List[List[Optional[float]]] = []
#     for q_smi, hits in zip(req.texts, smiles_hits):
#         try:
#             Gq = mol_to_nx(q_smi)
#         except ValueError:
#             ged_matrix.append([None] * len(hits))
#             continue

#         row_ged: List[Optional[float]] = []
#         for hit in hits:
#             if hit == "UNKNOWN" or Chem.MolFromSmiles(hit) is None:
#                 row_ged.append(None)
#                 continue
#             try:
#                 Gr = mol_to_nx(hit)
#             except ValueError:
#                 row_ged.append(None)
#                 continue

#             ged_val = nx.graph_edit_distance(
#                 Gq, Gr,
#                 node_ins_cost=lambda _: 1,
#                 node_del_cost=lambda _: 1,
#                 node_subst_cost=lambda a, b: 0 if a["label"] == b["label"] else 1,
#                 edge_ins_cost=lambda _: 1,
#                 edge_del_cost=lambda _: 1,
#                 timeout=300.0
#             )
#             row_ged.append(ged_val)

#         ged_matrix.append(row_ged)

#     # 5) Compute Hamming distances, skipping invalid SMILES
#     hamming_matrix: List[List[int]] = []
#     for q_smi, hits in zip(req.texts, smiles_hits):
#         mol_q = Chem.MolFromSmiles(q_smi)
#         if mol_q is None:
#             hamming_matrix.append([-1] * len(hits))
#             continue

#         fp_q = AllChem.GetMorganFingerprintAsBitVect(mol_q, 2, nBits=1024)
#         row_ham: List[int] = []
#         for hit in hits:
#             if hit == "UNKNOWN":
#                 row_ham.append(-1)
#                 continue

#             mol_h = Chem.MolFromSmiles(hit)
#             if mol_h is None:
#                 row_ham.append(-1)
#                 continue

#             fp_h = AllChem.GetMorganFingerprintAsBitVect(mol_h, 2, nBits=1024)
#             row_ham.append(hamming_distance(fp_q, fp_h))

#         hamming_matrix.append(row_ham)

#     # 6) Return response
#     return {
#         "ids": ids,
#         "distances": dists,
#         "smiles": smiles_hits,
#         "queries": req.texts,
#         "ged": ged_matrix,
#         "hamming": hamming_matrix,
#     }


@app.post("/add_to_index")
def add_to_index(req: AddToIndexRequest):
    if len(req.texts) != len(req.ids):
        raise HTTPException(status_code=400, detail="texts and ids must have the same length")
    embeddings = _embed_texts(req.texts)
    start_id = smiles_index.ntotal
    new_ids = np.arange(start_id, start_id + embeddings.shape[0], dtype="int64")
    smiles_index.add_with_ids(embeddings, new_ids)
    for faiss_id, mol_id in zip(new_ids, req.ids):
        smiles_id_map[int(faiss_id)] = mol_id
    faiss.write_index(smiles_index, SMILES_INDEX_PATH)
    with open(SMILES_IDMAP_PATH, "wb") as f:
        pickle.dump(smiles_id_map, f)
    return {"status": "success", "added": len(req.texts)}

@app.get("/list_index_ids")
def list_index_ids():
    return {"ids": list(smiles_id_map.values())}

@app.post("/complex_search")
async def complex_search(req: ComplexQueryRequest):
    parsed = parse_complex_query(req.query)
    if not parsed:
        raise HTTPException(status_code=400, detail="Could not parse complex query")
    intent = parsed.pop("intent")
    if intent == "find_inhibitors":
        target = parsed["target"]
        res = _search_vector(smiles_index, smiles_id_map, _embed_text(target), top_k=10)
        return {"intent": intent, "target": target, **res}
    elif intent == "similar_to":
        reference = parsed["reference"]
        res = _search_vector(smiles_index, smiles_id_map, _embed_text(reference), top_k=10)
        return {"intent": intent, "reference": reference, **res}
    else:
        raise HTTPException(status_code=400, detail="Unsupported intent")

# ─────────────── SELFIES Endpoints ───────────────

@app.post("/selfies_embed", response_model=EmbedResponse)
def selfies_embed(req: EmbedRequest):
    embeddings = _embed_texts(req.texts)
    return {"embeddings": embeddings.tolist()}

@app.post("/selfies_search", response_model=SelfiesSearchResponse)
def selfies_search(req: SearchRequest):
    vectors = np.array(req.vectors, dtype="float32")
    ids, dists = _search_index(selfies_index, selfies_id_map, vectors, req.top_k)
    selfies = get_selfies_for_ids(ids)
    return {"ids": ids, "distances": dists, "selfies": selfies, "queries": []}

@app.post("/selfies_semantic_search", response_model=SelfiesSearchResponse)
def selfies_semantic_search(req: SemanticSearchRequest):
    # 1) Build or retrieve normalized query embeddings
    query_vecs: List[np.ndarray] = []
    for selfie in req.texts:
        row = df_mols[df_mols["SELFIES"] == selfie]
        stored_vec = None
        if not row.empty:
            try:
                embedding_cols = [c for c in df_mols.columns if c.startswith("emb_")]
                vec = row[embedding_cols].values.astype("float32").flatten()
                if vec.size == 768 and not np.isnan(vec).any():
                    stored_vec = vec / np.linalg.norm(vec)
            except:
                pass
        fresh_vec = _embed_texts([selfie])[0].astype("float32")
        fresh_vec /= np.linalg.norm(fresh_vec)
        query_vecs.append(stored_vec if stored_vec is not None else fresh_vec)

    # 2) Run ANN search (fetch top_k + 1 to allow filtering out self-hits)
    mat = np.vstack(query_vecs)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    ids, dists = _search_index(selfies_index, selfies_id_map, mat, req.top_k + 1)

    # 3) Map back to SELFIES strings
    selfies_hits = get_selfies_for_ids(ids)

    # 3.5) FILTER OUT QUERY SELFIES FROM HITS AND TRUNCATE TO top_k
    filtered_ids = []
    filtered_dists = []
    filtered_selfies_hits = []
    for q_sf, id_row, dist_row, sf_row in zip(req.texts, ids, dists, selfies_hits):
        new_ids, new_dists, new_sfs = [], [], []
        for mid, dist, sf in zip(id_row, dist_row, sf_row):
            if sf != q_sf:
                new_ids.append(mid)
                new_dists.append(dist)
                new_sfs.append(sf)
            if len(new_ids) == req.top_k:
                break
        filtered_ids.append(new_ids)
        filtered_dists.append(new_dists)
        filtered_selfies_hits.append(new_sfs)

    # overwrite originals so steps 4–6 work unchanged
    ids = filtered_ids
    dists = filtered_dists
    selfies_hits = filtered_selfies_hits

    # 4) Compute GED matrix (SELFIES → SMILES → NetworkX)
    ged_matrix: List[List[Optional[float]]] = []
    for q_selfie, hit_batch in zip(req.texts, selfies_hits):
        q_row = df_mols[df_mols["SELFIES"] == q_selfie]
        if q_row.empty:
            ged_matrix.append([None] * len(hit_batch))
            continue

        # guard query graph construction
        try:
            Gq = mol_to_nx(q_row.iloc[0]["SMILES"])
        except ValueError:
            ged_matrix.append([None] * len(hit_batch))
            continue

        row_ged: List[Optional[float]] = []
        for hit_sf in hit_batch:
            if hit_sf == "UNKNOWN":
                row_ged.append(None)
                continue

            hit_row = df_mols[df_mols["SELFIES"] == hit_sf]
            if hit_row.empty:
                row_ged.append(None)
                continue

            # guard hit graph construction
            try:
                Gr = mol_to_nx(hit_row.iloc[0]["SMILES"])
            except ValueError:
                row_ged.append(None)
                continue

            ged_val = nx.graph_edit_distance(
                Gq, Gr,
                node_ins_cost=lambda _: 1,
                node_del_cost=lambda _: 1,
                node_subst_cost=lambda a, b: 0 if a["label"] == b["label"] else 1,
                edge_ins_cost=lambda _: 1,
                edge_del_cost=lambda _: 1,
                timeout=300.0
            )
            row_ged.append(ged_val)
        ged_matrix.append(row_ged)

    # 5) Compute Hamming distances for SELFIES hits
    hamming_matrix: List[List[int]] = []
    for q_selfie, hit_batch in zip(req.texts, selfies_hits):
        q_row = df_mols[df_mols["SELFIES"] == q_selfie]
        if q_row.empty:
            hamming_matrix.append([-1] * len(hit_batch))
            continue

        # guard query fingerprint generation
        try:
            mol_q = Chem.MolFromSmiles(q_row.iloc[0]["SMILES"])
            fp_q = AllChem.GetMorganFingerprintAsBitVect(mol_q, 2, nBits=1024)
        except Exception:
            hamming_matrix.append([-1] * len(hit_batch))
            continue

        row_ham: List[int] = []
        for hit_sf in hit_batch:
            if hit_sf == "UNKNOWN":
                row_ham.append(-1)
                continue

            hit_row = df_mols[df_mols["SELFIES"] == hit_sf]
            if hit_row.empty:
                row_ham.append(-1)
                continue

            # guard hit fingerprint generation
            try:
                mol_h = Chem.MolFromSmiles(hit_row.iloc[0]["SMILES"])
                fp_h = AllChem.GetMorganFingerprintAsBitVect(mol_h, 2, nBits=1024)
                ham = hamming_distance(fp_q, fp_h)
            except Exception:
                ham = -1

            row_ham.append(ham)
        hamming_matrix.append(row_ham)

    # 6) Return everything
    return {
        "ids": ids,
        "distances": dists,
        "selfies": selfies_hits,
        "queries": req.texts,
        "ged": ged_matrix,
        "hamming": hamming_matrix,
    }



# ─────────────── New SELFormer SELFIES Search ───────────────
@app.post("/selfies_semantic_search_SELFormer", response_model=SelfiesSearchResponse)
def selfies_semantic_search_SELFormer(req: SemanticSearchRequest):
    # 1) Build or retrieve normalized query embeddings
    query_vecs: List[np.ndarray] = []
    for selfie in req.texts:
        row = df_mols[df_mols["SELFIES"] == selfie]
        stored_vec = None
        if not row.empty:
            try:
                emb_cols = [c for c in df_mols.columns if c.startswith("emb_") and "SELFormer" in c]
                vec = row[emb_cols].values.astype("float32").flatten()
                if vec.size == sel_model.config.hidden_size and not np.isnan(vec).any():
                    stored_vec = vec / np.linalg.norm(vec)
            except:
                pass
        # compute fresh SELFormer embedding
        inputs = sel_tokenizer(
            [selfie], return_tensors="pt",
            padding=True, truncation=True,
            max_length=128
        ).to(DEVICE)
        with torch.no_grad():
            out = sel_model(**inputs)
            fresh = out.last_hidden_state.mean(dim=1).cpu().numpy()[0].astype("float32")
        fresh /= np.linalg.norm(fresh)
        query_vecs.append(stored_vec if stored_vec is not None else fresh)

    # 2) FAISS search (fetch top_k + 1 to allow filtering out self-hits)
    mat = np.vstack(query_vecs)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    ids, dists = _search_index(
        selfies_SELFormer_index,
        selfies_SELFormer_id_map,
        mat,
        req.top_k + 1
    )

    # 3) Map to SELFIES strings
    selfies_hits = get_selfies_for_ids(ids)

    # 3.5) FILTER OUT QUERY SELFIES FROM HITS AND TRUNCATE TO top_k
    filtered_ids = []
    filtered_dists = []
    filtered_selfies_hits = []
    for q_sf, id_row, dist_row, sf_row in zip(req.texts, ids, dists, selfies_hits):
        new_ids, new_dists, new_sfs = [], [], []
        for mid, dist, sf in zip(id_row, dist_row, sf_row):
            if sf != q_sf:
                new_ids.append(mid)
                new_dists.append(dist)
                new_sfs.append(sf)
            if len(new_ids) == req.top_k:
                break
        filtered_ids.append(new_ids)
        filtered_dists.append(new_dists)
        filtered_selfies_hits.append(new_sfs)

    # overwrite originals so steps 4–6 work unchanged
    ids = filtered_ids
    dists = filtered_dists
    selfies_hits = filtered_selfies_hits

    # 4) GED matrix
    ged_matrix: List[List[Optional[float]]] = []
    for q_sf, hit_batch in zip(req.texts, selfies_hits):
        q_row = df_mols[df_mols["SELFIES"] == q_sf]
        if q_row.empty:
            ged_matrix.append([None] * len(hit_batch))
            continue
        try:
            Gq = mol_to_nx(q_row.iloc[0]["SMILES"])
        except:
            ged_matrix.append([None] * len(hit_batch))
            continue
        row_ged: List[Optional[float]] = []
        for hit_sf in hit_batch:
            if hit_sf == "UNKNOWN":
                row_ged.append(None)
            else:
                hit_row = df_mols[df_mols["SELFIES"] == hit_sf]
                if hit_row.empty:
                    row_ged.append(None)
                    continue
                try:
                    Gr = mol_to_nx(hit_row.iloc[0]["SMILES"])
                    ged_val = nx.graph_edit_distance(
                        Gq, Gr,
                        node_ins_cost=lambda _: 1,
                        node_del_cost=lambda _: 1,
                        node_subst_cost=lambda a, b: 0 if a["label"] == b["label"] else 1,
                        edge_ins_cost=lambda _: 1,
                        edge_del_cost=lambda _: 1,
                        timeout=300.0
                    )
                    row_ged.append(ged_val)
                except:
                    row_ged.append(None)
        ged_matrix.append(row_ged)

    # 5) Hamming distances
    hamming_matrix: List[List[int]] = []
    for q_sf, hit_batch in zip(req.texts, selfies_hits):
        q_row = df_mols[df_mols["SELFIES"] == q_sf]
        if q_row.empty:
            hamming_matrix.append([-1] * len(hit_batch))
            continue
        try:
            mol_q = Chem.MolFromSmiles(q_row.iloc[0]["SMILES"])
            fp_q = AllChem.GetMorganFingerprintAsBitVect(mol_q, 2, nBits=1024)
        except:
            hamming_matrix.append([-1] * len(hit_batch))
            continue
        row_ham: List[int] = []
        for hit_sf in hit_batch:
            if hit_sf == "UNKNOWN":
                row_ham.append(-1)
            else:
                hit_row = df_mols[df_mols["SELFIES"] == hit_sf]
                if hit_row.empty:
                    row_ham.append(-1)
                    continue
                try:
                    mol_h = Chem.MolFromSmiles(hit_row.iloc[0]["SMILES"])
                    fp_h = AllChem.GetMorganFingerprintAsBitVect(mol_h, 2, nBits=1024)
                    row_ham.append(hamming_distance(fp_q, fp_h))
                except:
                    row_ham.append(-1)
        hamming_matrix.append(row_ham)

    # 6) Return everything
    return {
        "ids": ids,
        "distances": dists,
        "selfies": selfies_hits,
        "queries": req.texts,
        "ged": ged_matrix,
        "hamming": hamming_matrix,
    }


# @app.post("/selfies_semantic_search", response_model=SelfiesSearchResponse)
# def selfies_semantic_search(req: SemanticSearchRequest):
#     embeddings = _embed_texts(req.texts)
#     # Normalize embeddings to unit vectors
#     norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
#     ids, dists = _search_index(selfies_index, selfies_id_map, norm_embeddings, req.top_k)
#     selfies = get_selfies_for_ids(ids)
#     return {"ids": ids, "distances": dists, "selfies": selfies, "queries": req.texts}


@app.post("/selfies_add_to_index")
def selfies_add_to_index(req: AddToIndexRequest):
    if len(req.texts) != len(req.ids):
        raise HTTPException(status_code=400, detail="texts and ids must have the same length")
    embeddings = _embed_texts(req.texts)
    start_id = selfies_index.ntotal
    new_ids = np.arange(start_id, start_id + embeddings.shape[0], dtype="int64")
    selfies_index.add_with_ids(embeddings, new_ids)
    for faiss_id, mol_id in zip(new_ids, req.ids):
        selfies_id_map[int(faiss_id)] = mol_id
    faiss.write_index(selfies_index, SELFIES_INDEX_PATH)
    with open(SELFIES_IDMAP_PATH, "wb") as f:
        pickle.dump(selfies_id_map, f)
    return {"status": "success", "added": len(req.texts)}

@app.get("/selfies_list_index_ids")
def selfies_list_index_ids():
    return {"ids": list(selfies_id_map.values())}

def selfies_embed_text(text: str) -> np.ndarray:
    return _embed_text(text)

def selfies_search_vectors(vector: np.ndarray, top_k: int = 10) -> dict:
    return _search_vector(selfies_index, selfies_id_map, vector, top_k)

@app.post("/fingerprint", response_model=EmbedResponse)
def fingerprint(req: EmbedRequest):
    fps = []
    for smi in req.texts:
        try:
            mol = Chem.MolFromSmiles(smi)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            arr = np.zeros((1024,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        except:
            fps.append(np.zeros((1024,), dtype=np.float32))
    return {"embeddings": [fp.tolist() for fp in fps]}

@app.post("/fingerprint_search", response_model=SearchResponse)
def fingerprint_search(req: SearchRequest):
    vectors = np.array(req.vectors, dtype="float32")
    ids, dists = _search_index(fingerprint_index, fingerprint_id_map, vectors, req.top_k)

    smiles = []
    for row in ids:
        row_sm = []
        for mol_id in row:
            try:
                smi = df_mols[df_mols["CID"].astype(str) == str(mol_id)]["SMILES"].values
                row_sm.append(smi[0] if len(smi) > 0 else "UNKNOWN")
            except:
                row_sm.append("UNKNOWN")
        smiles.append(row_sm)

    return {"ids": ids, "distances": dists, "smiles": smiles, "queries": []}

@app.post("/fingerprint_semantic_search", response_model=SearchResponse)
def fingerprint_semantic_search(req: SemanticSearchRequest):
    # 1) Generate normalized fingerprint vectors for the query SMILES
    query_vecs = []
    for smi in req.texts:
        try:
            fp_vec = generate_fingerprint_vector(smi)  # normalized inside
            query_vecs.append(fp_vec)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error generating fingerprint for '{smi}': {e}")
    query_mat = np.vstack(query_vecs).astype("float32")

    # 2) FAISS cosine‐based search (fetch top_k + 1 to allow filtering out self-hits)
    ids, dists = _search_index(
        fingerprint_index,
        fingerprint_id_map,
        query_mat,
        req.top_k + 1
    )

    # 3) Map back to SMILES strings
    smiles_results: List[List[str]] = []
    for row in ids:
        row_sm = []
        for mol_id in row:
            vals = df_mols[df_mols["CID"].astype(str) == str(mol_id)]["SMILES"].values
            row_sm.append(vals[0] if len(vals) > 0 else "UNKNOWN")
        smiles_results.append(row_sm)

    # 3.5) FILTER OUT QUERY SMILES FROM HITS AND TRUNCATE TO top_k
    filtered_ids = []
    filtered_dists = []
    filtered_smiles = []
    for q_smi, id_row, dist_row, smi_row in zip(req.texts, ids, dists, smiles_results):
        new_ids, new_dists_row, new_smiles_row = [], [], []
        for mid, dist, sm in zip(id_row, dist_row, smi_row):
            if sm != q_smi:
                new_ids.append(mid)
                new_dists_row.append(dist)
                new_smiles_row.append(sm)
            if len(new_ids) == req.top_k:
                break
        filtered_ids.append(new_ids)
        filtered_dists.append(new_dists_row)
        filtered_smiles.append(new_smiles_row)

    # overwrite originals so steps 4–6 work unchanged
    ids = filtered_ids
    dists = filtered_dists
    smiles_results = filtered_smiles

    # 4) Compute GED matrix (unit costs, 2-minute timeout)
    ged_matrix: List[List[Optional[float]]] = []
    for q_smi, hits in zip(req.texts, smiles_results):
        Gq = mol_to_nx(q_smi)
        row_ged: List[Optional[float]] = []
        for hit_smi in hits:
            if hit_smi == "UNKNOWN":
                row_ged.append(None)
            else:
                Gr = mol_to_nx(hit_smi)
                ged_val = nx.graph_edit_distance(
                    Gq, Gr,
                    node_ins_cost=lambda _: 1,
                    node_del_cost=lambda _: 1,
                    node_subst_cost=lambda a, b: 0 if a['label'] == b['label'] else 1,
                    edge_ins_cost=lambda _: 1,
                    edge_del_cost=lambda _: 1,
                    timeout=300.0
                )
                row_ged.append(ged_val)
        ged_matrix.append(row_ged)

    # 5) Compute Hamming distances on original bit fingerprints
    hamming_matrix: List[List[int]] = []
    for smi, hits in zip(req.texts, smiles_results):
        fp_query = smiles_to_fp.get(smi)
        row_ham: List[int] = []
        for hit_smi in hits:
            if fp_query is None or hit_smi == "UNKNOWN":
                row_ham.append(-1)
            else:
                fp_hit = smiles_to_fp.get(hit_smi)
                row_ham.append(hamming_distance(fp_query, fp_hit) if fp_hit is not None else -1)
        hamming_matrix.append(row_ham)

    # 6) Return everything, preserving cosine ranking
    return {
        "ids": ids,
        "distances": dists,
        "smiles": smiles_results,
        "queries": req.texts,
        "ged": ged_matrix,
        "hamming": hamming_matrix,
    }





@app.post("/fingerprint_semantic_search_old", response_model=SearchResponse)
def fingerprint_semantic_search(req: SemanticSearchRequest):
    query_vecs = []
    for smi in req.texts:
        try:
            fp_vec = generate_fingerprint_vector(smi)
            if fp_vec is None:
                raise ValueError("Failed to generate fingerprint")
            fp_vec = fp_vec.astype("float32")
            fp_vec /= np.linalg.norm(fp_vec)
            query_vecs.append(fp_vec)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error generating fingerprint for '{smi}': {e}")

    query_vecs_np = np.vstack(query_vecs)
    ids, dists = _search_index(fingerprint_index, fingerprint_id_map, query_vecs_np, req.top_k)

    smiles_results = []
    for row in ids:
        row_sm = []
        for mol_id in row:
            try:
                smi = df_mols[df_mols["CID"].astype(str) == str(mol_id)]["SMILES"].values
                row_sm.append(smi[0] if len(smi) > 0 else "UNKNOWN")
            except:
                row_sm.append("UNKNOWN")
        smiles_results.append(row_sm)

    return {
        "ids": ids,
        "distances": dists,
        "smiles": smiles_results,
        "queries": req.texts
    }

@app.post("/fingerprint_semantic_search_Tanimoto", response_model=SearchResponse)
def fingerprint_semantic_search(req: SemanticSearchRequest):
    final_ids, final_smiles, final_sims = [], [], []

    # 1) Tanimoto ranking (over-fetch top_k + 1 to allow filtering)
    for smi in req.texts:
        fp = smiles_to_fp.get(smi)
        if fp is None:
            raise HTTPException(400, f"SMILES not found: {smi}")

        sims = DataStructs.BulkTanimotoSimilarity(fp, all_fps)
        # fetch one extra to allow removing the query itself
        idx_sorted = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:req.top_k + 1]

        # initial lists
        ids_row = [str(df_mols.iloc[i]["CID"]) for i in idx_sorted]
        smi_row = [df_mols.iloc[i]["SMILES"] for i in idx_sorted]
        sim_row = [sims[i] for i in idx_sorted]

        # 1.5) Filter out the query SMILES and truncate to top_k
        new_ids, new_smiles, new_sims = [], [], []
        for cid, sm_hit, sim_val in zip(ids_row, smi_row, sim_row):
            if sm_hit != smi:
                new_ids.append(cid)
                new_smiles.append(sm_hit)
                new_sims.append(sim_val)
            if len(new_ids) == req.top_k:
                break

        final_ids.append(new_ids)
        final_smiles.append(new_smiles)
        final_sims.append(new_sims)

    # 2) Compute GED matrix with 2-minute timeout
    ged_matrix: List[List[Optional[float]]] = []
    for q_smi, hits in zip(req.texts, final_smiles):
        Gq = mol_to_nx(q_smi)
        row_ged: List[Optional[float]] = []
        for hit_smi in hits:
            if hit_smi == "UNKNOWN":
                row_ged.append(None)
            else:
                Gr = mol_to_nx(hit_smi)
                ged_val = nx.graph_edit_distance(
                    Gq, Gr,
                    node_ins_cost=lambda _: 1,
                    node_del_cost=lambda _: 1,
                    node_subst_cost=lambda a, b: 0 if a['label'] == b['label'] else 1,
                    edge_ins_cost=lambda _: 1,
                    edge_del_cost=lambda _: 1,
                    timeout=300.0
                )
                row_ged.append(ged_val)
        ged_matrix.append(row_ged)

    # 3) Compute Hamming distances (edit cost on bit vectors)
    hamming_matrix: List[List[int]] = []
    for smi, hits in zip(req.texts, final_smiles):
        fp_query = smiles_to_fp.get(smi)
        row_ham: List[int] = []
        for hit_smi in hits:
            if fp_query is None or hit_smi == "UNKNOWN":
                row_ham.append(-1)
            else:
                fp_hit = smiles_to_fp.get(hit_smi)
                ham = hamming_distance(fp_query, fp_hit) if fp_hit is not None else -1
                row_ham.append(ham)
        hamming_matrix.append(row_ham)

    # 4) Return combined results (ranking by Tanimoto preserved)
    return {
        "ids": final_ids,
        "distances": final_sims,
        "smiles": final_smiles,
        "queries": req.texts,
        "ged": ged_matrix,
        "hamming": hamming_matrix,
    }
@app.post("/fingerprints_Tanimoto_investigate", response_model=SearchResponse)
def fingerprints_Tanimoto_investigate(req: SemanticSearchRequest):
    final_ids, final_smiles, final_sims = [], [], []

    # Prepare storage for float-version fingerprints
    query_fp_floats: List[List[float]] = []
    hit_fp_floats: List[List[List[float]]] = []

    # 1) Tanimoto-based similarity search (over-fetch top_k + 1 to allow filtering)
    for smi in req.texts:
        # compute and store float fingerprint for query
        vec_q = generate_fingerprint_vector(smi)
        query_fp_floats.append(vec_q.tolist() if vec_q is not None else [])

        fp = smiles_to_fp.get(smi)
        if fp is None:
            raise HTTPException(status_code=400, detail=f"SMILES not found: {smi}")

        sims = DataStructs.BulkTanimotoSimilarity(fp, all_fps)
        idx_sorted = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:req.top_k + 1]

        ids_row = [str(df_mols.iloc[i]["CID"]) for i in idx_sorted]
        smi_row = [df_mols.iloc[i]["SMILES"] for i in idx_sorted]
        sim_row = [sims[i] for i in idx_sorted]

        # filter out self-hit and truncate
        new_ids, new_smiles, new_sims, row_fp_floats = [], [], [], []
        for cid, hit_smi, sim_val in zip(ids_row, smi_row, sim_row):
            if hit_smi != smi and len(new_ids) < req.top_k:
                new_ids.append(cid)
                new_smiles.append(hit_smi)
                new_sims.append(sim_val)
                # compute float fingerprint for hit
                vec_h = generate_fingerprint_vector(hit_smi)
                row_fp_floats.append(vec_h.tolist() if vec_h is not None else [])
        final_ids.append(new_ids)
        final_smiles.append(new_smiles)
        final_sims.append(new_sims)
        hit_fp_floats.append(row_fp_floats)

    # 2) Compute GED matrix (unit costs, 5-min timeout)
    ged_matrix: List[List[Optional[float]]] = []
    for q_smi, hits in zip(req.texts, final_smiles):
        Gq = mol_to_nx(q_smi)
        row_ged: List[Optional[float]] = []
        for hit_smi in hits:
            if hit_smi == "UNKNOWN":
                row_ged.append(None)
            else:
                Gr = mol_to_nx(hit_smi)
                ged_val = nx.graph_edit_distance(
                    Gq, Gr,
                    node_ins_cost=lambda _: 1,
                    node_del_cost=lambda _: 1,
                    node_subst_cost=lambda a, b: 0 if a['label'] == b['label'] else 1,
                    edge_ins_cost=lambda _: 1,
                    edge_del_cost=lambda _: 1,
                    timeout=300.0
                )
                row_ged.append(ged_val)
        ged_matrix.append(row_ged)

    # 3) Compute Hamming distances
    hamming_matrix: List[List[int]] = []
    for smi, hits in zip(req.texts, final_smiles):
        fp_query = smiles_to_fp.get(smi)
        row_ham: List[int] = []
        for hit_smi in hits:
            if fp_query is None or hit_smi == "UNKNOWN":
                row_ham.append(-1)
            else:
                fp_hit = smiles_to_fp.get(hit_smi)
                row_ham.append(hamming_distance(fp_query, fp_hit) if fp_hit is not None else -1)
        hamming_matrix.append(row_ham)

    # 4) Compute Cosine similarity between float fingerprints
    cosine_matrix: List[List[float]] = []
    for q_vec, hits in zip(query_fp_floats, hit_fp_floats):
        q_arr = np.array(q_vec, dtype=float)
        q_norm = np.linalg.norm(q_arr)
        row_cos: List[float] = []
        for h_vec in hits:
            h_arr = np.array(h_vec, dtype=float)
            denom = q_norm * np.linalg.norm(h_arr)
            row_cos.append(float(np.dot(q_arr, h_arr) / denom) if denom else 0.0)
        cosine_matrix.append(row_cos)

    # 5) Collect raw bit fingerprints
    query_fp_bits: List[Optional[List[int]]] = []
    for smi in req.texts:
        fp = smiles_to_fp.get(smi)
        query_fp_bits.append(list(fp) if fp is not None else None)

    hit_fp_bits: List[List[Optional[List[int]]]] = []
    for hits in final_smiles:
        row_bits: List[Optional[List[int]]] = []
        for hit_smi in hits:
            fp = smiles_to_fp.get(hit_smi)
            row_bits.append(list(fp) if fp is not None else None)
        hit_fp_bits.append(row_bits)

    # 6) Return combined response
    return {
        "ids": final_ids,
        "distances": final_sims,
        "smiles": final_smiles,
        "queries": req.texts,
        "fingerprints_query": query_fp_bits,
        "fingerprints_hits": hit_fp_bits,
        "fingerprints_query_floats": query_fp_floats,
        "fingerprints_hits_floats": hit_fp_floats,
        "ged": ged_matrix,
        "hamming": hamming_matrix,
        "tanimoto": final_sims,
        "cosine": cosine_matrix,
    }


@app.post("/fingerprints_cosine_investigate", response_model=SearchResponse)
def fingerprint_semantic_search(req: SemanticSearchRequest):
    # 1) Generate normalized fingerprint vectors for the query SMILES
    query_vecs = []
    for smi in req.texts:
        try:
            fp_vec = generate_fingerprint_vector(smi)  # normalized inside
            query_vecs.append(fp_vec)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error generating fingerprint for '{smi}': {e}")
    query_mat = np.vstack(query_vecs).astype("float32")

    # Save the float-version fingerprints for queries
    query_fp_floats: List[List[float]] = [vec.tolist() for vec in query_vecs]

    # 2) FAISS cosine‐based search (fetch top_k + 1 to allow filtering out self-hits)
    ids, dists = _search_index(
        fingerprint_index,
        fingerprint_id_map,
        query_mat,
        req.top_k + 1
    )

    # 3) Map back to SMILES strings
    smiles_results: List[List[str]] = []
    for row in ids:
        row_sm = []
        for mol_id in row:
            vals = df_mols[df_mols["CID"].astype(str) == str(mol_id)]["SMILES"].values
            row_sm.append(vals[0] if len(vals) > 0 else "UNKNOWN")
        smiles_results.append(row_sm)

    # 3.5) FILTER OUT QUERY SMILES FROM HITS AND TRUNCATE TO top_k
    filtered_ids, filtered_dists, filtered_smiles = [], [], []
    for q_smi, id_row, dist_row, smi_row in zip(req.texts, ids, dists, smiles_results):
        new_ids, new_dists, new_smiles = [], [], []
        for mid, dist, sm in zip(id_row, dist_row, smi_row):
            if sm != q_smi:
                new_ids.append(mid)
                new_dists.append(dist)
                new_smiles.append(sm)
            if len(new_ids) == req.top_k:
                break
        filtered_ids.append(new_ids)
        filtered_dists.append(new_dists)
        filtered_smiles.append(new_smiles)

    ids = filtered_ids
    dists = filtered_dists
    smiles_results = filtered_smiles

    # 3.6) Gather raw ECFP4 fingerprints as bit lists
    query_fp_bits: List[Optional[List[int]]] = []
    for smi in req.texts:
        fp = smiles_to_fp.get(smi)
        query_fp_bits.append(list(fp) if fp is not None else None)

    hit_fp_bits: List[List[Optional[List[int]]]] = []
    for hits in smiles_results:
        row_bits: List[Optional[List[int]]] = []
        for hit_smi in hits:
            fp = smiles_to_fp.get(hit_smi)
            row_bits.append(list(fp) if fp is not None else None)
        hit_fp_bits.append(row_bits)

    # Collect float-version fingerprints for hits
    hit_fp_floats: List[List[List[float]]] = []
    for hits in smiles_results:
        row_floats: List[List[float]] = []
        for hit_smi in hits:
            try:
                vec = generate_fingerprint_vector(hit_smi)
                row_floats.append(vec.tolist())
            except:
                row_floats.append([])
        hit_fp_floats.append(row_floats)

    # 4) Compute GED matrix (unit costs, 2-minute timeout)
    ged_matrix: List[List[Optional[float]]] = []
    for q_smi, hits in zip(req.texts, smiles_results):
        Gq = mol_to_nx(q_smi)
        row_ged: List[Optional[float]] = []
        for hit_smi in hits:
            if hit_smi == "UNKNOWN":
                row_ged.append(None)
            else:
                Gr = mol_to_nx(hit_smi)
                ged_val = nx.graph_edit_distance(
                    Gq, Gr,
                    node_ins_cost=lambda _: 1,
                    node_del_cost=lambda _: 1,
                    node_subst_cost=lambda a, b: 0 if a['label'] == b['label'] else 1,
                    edge_ins_cost=lambda _: 1,
                    edge_del_cost=lambda _: 1,
                    timeout=120.0
                )
                row_ged.append(ged_val)
        ged_matrix.append(row_ged)

    # 5) Compute Hamming distances
    hamming_matrix: List[List[int]] = []
    for smi, hits in zip(req.texts, smiles_results):
        fp_q = smiles_to_fp.get(smi)
        row_ham: List[int] = []
        for hit in hits:
            if fp_q is None or hit == "UNKNOWN":
                row_ham.append(-1)
            else:
                fp_h = smiles_to_fp.get(hit)
                row_ham.append(hamming_distance(fp_q, fp_h) if fp_h is not None else -1)
        hamming_matrix.append(row_ham)

    # 6) Compute Tanimoto coefficients
    tanimoto_matrix: List[List[float]] = []
    for smi, hits in zip(req.texts, smiles_results):
        fp_q = smiles_to_fp.get(smi)
        row_tan: List[float] = []
        if fp_q is None:
            row_tan = [-1.0] * len(hits)
        else:
            fps_hits = [smiles_to_fp.get(hit) for hit in hits]
            sims = []
            for fp_h in fps_hits:
                if fp_h is None:
                    sims.append(-1.0)
                else:
                    sims.append(float(DataStructs.TanimotoSimilarity(fp_q, fp_h)))
            row_tan = sims
        tanimoto_matrix.append(row_tan)

    # 7) Return full response
    return {
        "ids": ids,
        "distances": dists,
        "smiles": smiles_results,
        "queries": req.texts,
        "fingerprints_query": query_fp_bits,
        "fingerprints_hits": hit_fp_bits,
        "fingerprints_query_floats": query_fp_floats,
        "fingerprints_hits_floats": hit_fp_floats,
        "ged": ged_matrix,
        "hamming": hamming_matrix,
        "tanimoto": tanimoto_matrix
    }



@app.post("/fingerprint_semantic_search_Tanimoto_old", response_model=SearchResponse)
def fingerprint_semantic_search(req: SemanticSearchRequest):
    final_ids, final_smiles, final_sims = [], [], []
    for smi in req.texts:
        fp = smiles_to_fp.get(smi)
        if fp is None:
            raise HTTPException(400, f"SMILES not found: {smi}")
        sims = DataStructs.BulkTanimotoSimilarity(fp, all_fps)
        idx_sorted = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:req.top_k]
        ids_row = [ str(df_mols.iloc[i]["CID"]) for i in idx_sorted ]
        smi_row = [ df_mols.iloc[i]["SMILES"]   for i in idx_sorted ]
        sim_row = [ sims[i]                      for i in idx_sorted ]
        final_ids.append(ids_row)
        final_smiles.append(smi_row)
        final_sims.append(sim_row)
    return {
        "ids": final_ids,
        "distances": final_sims,
        "smiles": final_smiles,
        "queries": req.texts
    }
# ───────── SELFIES FINGERPRINT GENERATION ────────────────────
@app.post("/selfies_fingerprint", response_model=EmbedResponse)
def selfies_fingerprint(req: EmbedRequest):
    fps = []
    for sf_str in req.texts:
        try:
            # decode SELFIES → SMILES
            smi = sf.decoder(sf_str)
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SELFIES: {sf_str}")
            # generate 1024‐bit Morgan fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            arr = np.zeros((1024,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        except Exception:
            fps.append(np.zeros((1024,), dtype=np.float32))
    return {"embeddings": [fp.tolist() for fp in fps]}


# ───────── SELFIES FINGERPRINT SEARCH ───────────────────────
@app.post("/selfies_fingerprint_search", response_model=SelfiesSearchResponse)
def selfies_fingerprint_search(req: SearchRequest):
    vectors = np.array(req.vectors, dtype="float32")
    ids, dists = _search_index(selfies_fp_index, selfies_fp_id_map, vectors, req.top_k)

    selfies_out = []
    for row in ids:
        row_sf = []
        for mol_id in row:
            try:
                # lookup SELFIES by CID in your df or selfies_map
                sf_vals = df_mols[df_mols["CID"].astype(str) == str(mol_id)]["SELFIES"].values
                row_sf.append(sf_vals[0] if len(sf_vals) > 0 else "UNKNOWN")
            except:
                row_sf.append("UNKNOWN")
        selfies_out.append(row_sf)

    return {"ids": ids, "distances": dists, "selfies": selfies_out, "queries": []}


# ───────── SELFIES FINGERPRINT SEMANTIC SEARCH ──────────────
@app.post("/selfies_fingerprint_semantic_search", response_model=SelfiesSearchResponse)
def selfies_fingerprint_semantic_search(req: SemanticSearchRequest):
    query_fps = []
    for sf_str in req.texts:
        try:
            smi = sf.decoder(sf_str)
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SELFIES: {sf_str}")
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            arr = np.zeros((1024,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            # normalize
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr /= norm
            query_fps.append(arr)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error generating fingerprint for '{sf_str}': {e}")

    query_mat = np.vstack(query_fps).astype("float32")
    ids, dists = _search_index(selfies_fp_index, selfies_fp_id_map, query_mat, req.top_k)

    selfies_out = []
    for row in ids:
        row_sf = []
        for mol_id in row:
            try:
                sf_vals = df_mols[df_mols["CID"].astype(str) == str(mol_id)]["SELFIES"].values
                row_sf.append(sf_vals[0] if len(sf_vals) > 0 else "UNKNOWN")
            except:
                row_sf.append("UNKNOWN")
        selfies_out.append(row_sf)

    return {"ids": ids, "distances": dists, "selfies": selfies_out, "queries": req.texts}
