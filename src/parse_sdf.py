import os
import gzip
import shutil
import csv
import logging
import time
from rdkit import Chem
import selfies
import torch
from transformers import AutoTokenizer, AutoModel

# ─── PATH SETUP ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
#DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))
DATA_DIR = "/mnt/d/akarmark/data"

SDF_FILENAME = "Compound_000000001_000500000.sdf"
GZ_FILENAME = SDF_FILENAME + ".gz"

sdf_path = os.path.join(DATA_DIR, SDF_FILENAME)
gz_path = os.path.join(DATA_DIR, GZ_FILENAME)

if not os.path.exists(sdf_path) and os.path.exists(gz_path):
    print(f"– Decompressing {GZ_FILENAME} → {SDF_FILENAME}")
    with gzip.open(gz_path, "rb") as f_in, open(sdf_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

# ─── LOGGING SETUP ────────────────────────────────────────────────────────────
LOG_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "logs"))
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("parse_sdf")
logger.setLevel(logging.ERROR)
file_handler = logging.FileHandler(os.path.join(LOG_DIR, "invalidMols.log"))
file_handler.setFormatter(logging.Formatter("%(asctime)s – %(levelname)s – %(message)s"))
logger.addHandler(file_handler)
logger.propagate = False

perf_logger = logging.getLogger("performance")
perf_logger.setLevel(logging.INFO)
perf_handler = logging.FileHandler(os.path.join(LOG_DIR, "performanceMetrics.log"))
perf_handler.setFormatter(logging.Formatter("%(asctime)s – %(message)s"))
perf_logger.addHandler(perf_handler)
perf_logger.propagate = False

# ─── OUTPUT CSV ──────────────────────────────────────────────────────────────
CSV_OUT = os.path.normpath(os.path.join(DATA_DIR, "sample_converted.csv"))

# ─── LOAD CHEMBERTA ──────────────────────────────────────────────────────────
# print("🔄 Loading ChemBERTa model...")
# tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
# model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
# model.eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

print("🔄 Loading ChemBERTa model…")
tokenizer = AutoTokenizer.from_pretrained(
    "seyonec/ChemBERTa-zinc-base-v1",
    use_safetensors=True          # force safetensors format if available
)
model = AutoModel.from_pretrained(
    "seyonec/ChemBERTa-zinc-base-v1",
    use_safetensors=True
)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# ─── BATCH SIZE ───────────────────────────────────────────────────────────────
BATCH_SIZE = 32  # Tune this depending on your memory and CPU

def get_chemberta_batch_embeddings(smiles_list):
    try:
        inputs = tokenizer(smiles_list, return_tensors="pt", truncation=True,
                           padding=True, max_length=128)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            embeddings = last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"Batch embedding error: {e}")
        return [None] * len(smiles_list)

# ─── PARSE FUNCTION ──────────────────────────────────────────────────────────
def parse_sdf_to_csv():
    start_time = time.time()

    suppl = Chem.SDMolSupplier(sdf_path)
    valid_mol_count = 0
    invalid_mol_count = 0
    buffer = []

    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["CID", "SMILES", "SELFIES", "ChemBERTa_Embedding"])

        for idx, mol in enumerate(suppl):
            if mol is None:
                invalid_mol_count += 1
                logger.error(f"Invalid molecule at index {idx}")
                continue

            try:
                cid = mol.GetProp("PUBCHEM_COMPOUND_CID")
                smiles = Chem.MolToSmiles(mol)
                selfies_str = selfies.encoder(smiles)
                buffer.append((cid, smiles, selfies_str))

                if len(buffer) == BATCH_SIZE:
                    cids, smiles_batch, selfies_batch = zip(*buffer)
                    embeddings = get_chemberta_batch_embeddings(list(smiles_batch))
                    for cid, smi, sfs, emb in zip(cids, smiles_batch, selfies_batch, embeddings):
                        if emb is None:
                            logger.error(f"Embedding failed for CID {cid}")
                            invalid_mol_count += 1
                            continue
                        embedding_str = ";".join(f"{x:.4f}" for x in emb)
                        writer.writerow([cid, smi, sfs, embedding_str])
                        valid_mol_count += 1
                    buffer = []

            except Exception as e:
                logger.error(f"Error at index {idx}: {e}")
                invalid_mol_count += 1

        # Final flush
        if buffer:
            cids, smiles_batch, selfies_batch = zip(*buffer)
            embeddings = get_chemberta_batch_embeddings(list(smiles_batch))
            for cid, smi, sfs, emb in zip(cids, smiles_batch, selfies_batch, embeddings):
                if emb is None:
                    logger.error(f"Embedding failed for CID {cid}")
                    invalid_mol_count += 1
                    continue
                embedding_str = ";".join(f"{x:.4f}" for x in emb)
                writer.writerow([cid, smi, sfs, embedding_str])
                valid_mol_count += 1

    # ─── PERFORMANCE METRICS ────────────────────────────────────────────────
    end_time = time.time()
    total_time = end_time - start_time
    total_mols = valid_mol_count + invalid_mol_count
    avg_time_per_mol = total_time / total_mols if total_mols > 0 else 0
    mols_per_sec = total_mols / total_time if total_time > 0 else 0

    print(f"✅ Done. Wrote {CSV_OUT}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Avg time per molecule: {avg_time_per_mol:.4f} s")
    print(f"Molecules/sec: {mols_per_sec:.2f}")
    print(f"Invalid molecules: {invalid_mol_count} of {total_mols}")

    perf_logger.info(f"Parsed file: {SDF_FILENAME}")
    perf_logger.info(f"Total molecules processed: {total_mols}")
    perf_logger.info(f"Valid: {valid_mol_count}, Invalid: {invalid_mol_count}")
    perf_logger.info(f"Total time: {total_time:.2f} seconds")
    perf_logger.info(f"Avg time per molecule: {avg_time_per_mol:.4f} s")
    perf_logger.info(f"Molecules per second: {mols_per_sec:.2f}")

    # ─── SAVE CLEAN SMILES FILE ──────────────────────────────────────────────
    clean_csv_path = os.path.join(DATA_DIR, "sample_smiles.csv")
    try:
        with open(CSV_OUT, "r") as original_file, open(clean_csv_path, "w", newline="") as clean_file:
            reader = csv.DictReader(original_file)
            writer = csv.DictWriter(clean_file, fieldnames=["CID", "SMILES"])
            writer.writeheader()
            for row in reader:
                writer.writerow({"CID": row["CID"], "SMILES": row["SMILES"]})
        print(f"🧼 Clean SMILES CSV written to {clean_csv_path}")
    except Exception as e:
        logger.error(f"Failed to generate clean smiles CSV: {e}")

# ─── RUN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parse_sdf_to_csv()
