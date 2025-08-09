# MolSimSearch

## Overview

This repository evaluates molecular similarity search methods by comparing transformer-based embeddings (SMILES and SELFIES) from ChemBERTa and SELFormer against 1024‑bit ECFP4 fingerprints. Using a set of chemically diverse query molecules, the code retrieves top‑k neighbours (k ∈ {5, 8, 10, 15, 20, 25, 35}) and evaluates hits on **chemical similarity** (Hamming distance) and **structural similarity** (Graph Edit Distance, GED). Statistical significance is assessed using one‑way ANOVA and Tukey's HSD. The experiments show that ECFP4/Tanimoto gives the best performance; SELFIES embeddings outperform SMILES embeddings but remain inferior to fingerprints.

---

## Quick features

* Download and parse large collections of molecules from PubChem (SDF).
* Generate SMILES/SELFIES representations.
* Produce transformer embeddings (ChemBERTa & SELFormer) for SMILES and SELFIES.
* Produce 1024‑bit ECFP4 fingerprints.
* Ingest embeddings and fingerprints into FAISS (5 indexes: chemBERTa SMILES, chemBERTa SELFIES, SELFormer SMILES, SELFormer SELFIES, fingerprints).
* Run top‑k retrieval and compute Hamming distance, GED and summary statistics.
* Generate plots of mean GED and mean Hamming vs. k and export CSV summaries.

---

## Prerequisites

* Python 3.8+
* Enough disk space for the downloaded molecule set (the scripts are written to download \~500K molecules).
* Access to a machine with enough RAM/CPU for embedding generation (GPU recommended for transformer inference but not required).

## Repository structure (important files in `src/`)

```
src/
├─ app.py                       # FastAPI/Uvicorn entry point used by the pipeline
├─ select_diverse.py            # Creates a queries.txt of chemically diverse molecules
├─ download_sdf.py              # Downloads molecules from PubChem (SDF)
├─ parse_sdf.py                 # Extracts SMILES and SELFIES into CSV
├─ generate_embeddings.py       # ChemBERTa embeddings (SMILES)
├─ generate_selfies_embeddings.py # ChemBERTa embeddings (SELFIES)
├─ generate_embeddings_SELFormer.py # SELFormer embeddings (SMILES)
├─ generate_selfies_embeddings_SELFormer.py # SELFormer embeddings (SELFIES)
├─ generate_fingerprints_SMILES.py # ECFP4 fingerprint generation
├─ ingest_faiss.py              # Insert chemBERTa embeddings into FAISS
├─ ingest_faiss_SELFormer.py    # Insert SELFormer embeddings into FAISS
├─ ingest_raw_fingerprints.py   # Insert fingerprints into FAISS
├─ run_full_pipeline.py         # Runs the full retrieval + evaluation pipeline
└─ utils/                       # Utility helpers (parsing, GED, Hamming, stats)
```

---

## Data preparation (one‑time setup)

> **Note:** This is a one‑time run. Once embeddings and fingerprints are generated and inserted into FAISS, you can skip these steps for subsequent experiments.

Install libraries:
```
pip install rdkit
pip install selfies
pip install torch
pip install transformers
pip install pandas
pip install faiss-cpu
pip install matplotlib
```

1. `cd src`
2. Download molecules from PubChem:

```bash
python download_sdf.py
```

3. Parse SDF and produce SMILES/SELFIES CSV:

```bash
python parse_sdf.py
```

4. Generate embeddings and fingerprints (order shown below):

```bash
python generate_embeddings.py
python generate_selfies_embeddings.py
python generate_embeddings_SELFormer.py
python generate_selfies_embeddings_SELFormer.py
python generate_fingerprints_SMILES.py

```

5. Ingest into FAISS (creates 5 indexes: chemBERTa SMILES/SELFIES, SELFormer SMILES/SELFIES, fingerprints):

```bash
python ingest_faiss.py
python ingest_faiss_SELFormer.py
python ingest_raw_fingerprints.py
```

After these steps you should have 5 FAISS indexes populated and be ready to run experiments.

---

## Generating chemically diverse queries

If you don't want to manually pick query molecules, the repository includes a script that selects chemically diverse queries using a MinMax algorithm (based on Tanimoto coefficient) and your precomputed fingerprints.

```bash
cd src
python select_diverse.py
```

This produces a `queries.txt` file containing 20 chemically diverse query SMILES (one per line). Assumes fingerprints are already available in CSV form.

---

## Running the Molecular Similarity Search (end‑to‑end)

Assumptions:

* FAISS indexes are built and available.
* You have a `queries.txt` file with query SMILES (one SMILES per line, **do not** enclose in quotes).

Steps:
<!--
1. From the repository root (the folder just outside `src`) start the app:

```bash
uvicorn src.app:app --reload
```
-->
2. In a new shell, `cd src` and run the pipeline:

```bash
cd src
python run_full_pipeline.py
```

When the run completes you will find the following output files:

* `results_full.csv` — top‑k hits information for each query and method
* `ged_summary.csv` — mean/median/std summary statistics for GED
* `hamming_summary.csv` — mean/median/std summary statistics for Hamming distance
* `mean_GED_vs_k.png` — line plot of mean GED vs. number of hits for each method
* `mean_Hamming_vs_k.png` — line plot of mean Hamming vs. number of hits for each method

---

## Expected evaluation metrics & statistics

* **Similarity measures used for retrieval**

  * *Embeddings:* Cosine similarity
  * *Fingerprints:* Tanimoto coefficient

* **Post‑retrieval evaluation**

  * *Hamming distance* (chemical similarity on fingerprint bit vectors)
  * *Graph Edit Distance (GED)* (structural similarity)

* **Statistical testing**

  * One‑way ANOVA across methods
  * Tukey's HSD for pairwise comparisons

**Key experimental finding (from the paper):** ECFP4/Tanimoto yields the lowest mean Hamming distances (mean = 12.5 at k = 35) and lowest mean GEDs (mean = 18.9). SELFIES embeddings outperform SMILES embeddings but are still inferior to fingerprints.

---

## Queries file format (important)

* Plain text file named exactly `queries.txt`
* One SMILES string per line
* DO NOT wrap SMILES in quotes
* Save and close the file before running `run_full_pipeline.py`

---

## Troubleshooting & tips

* If embedding generation is slow, consider using a GPU or batching inputs.
* RDKit installation can be simplified using `conda`:
  `conda install -c conda-forge rdkit`
* FAISS installation: use `faiss-cpu` for CPU-only environments, or `faiss-gpu` if you have CUDA.
* Make sure the FAISS index paths in the ingest scripts match your local paths.
* If plots are not generated, check that `matplotlib` is installed and that your run completed without exceptions.

---




