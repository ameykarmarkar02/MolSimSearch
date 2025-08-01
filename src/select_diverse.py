#!/usr/bin/env python3
import pandas as pd
import numpy as np

# STEP 1: Paths to your two CSVs
FP_CSV     = '/mnt/d/akarmark/data/smiles_fingerprints.csv'  # has CID, fp_0…fp_1023
SMILES_CSV = '/mnt/d/akarmark/data/sample_smiles.csv'       # has CID, SMILES

# STEP 2: Load data
df_fp     = pd.read_csv(FP_CSV, dtype={'CID':str})
df_smiles = pd.read_csv(SMILES_CSV, usecols=['CID','SMILES'], dtype=str)

# STEP 3: Merge on CID to recover SMILES alongside fps
df = pd.merge(df_fp, df_smiles, on='CID', how='inner')

# Normalize SMILES for filtering
df['SMILES'] = df['SMILES'].str.strip()

# STEP 4: Exclusion list (same 30 SMILES)
exclude = {
    "c1ccc2cc3c(ccc4cccnc43)cc2c1",
    "O=C1C(C2CCCCC2)C(=O)N(C2CCCCC2)C(=O)N1c1ccccc1",
    "[CH3][Sb]([CH3])([CH3])([Cl])[Cl]",
    "CC(=O)N(Cc1ccccc1)N=O",
    "O=[N+]([O-])[O-].c1ccc([I+]c2ccccc2)cc1",
    "OCCCCCCCCCCCCF",
    "CP(C)Cl",
    "C=CCCCCCCCCCCCCCCCCCCCCCC",
    "CCCCCCCCCCCCCCCCCC(=O)OCCOCCO",
    "CCCCCCCCCCCC1CCC(=O)O1",
    "CCN(CCCN1c2ccccc2Sc2ccc(C(F)(F)(F))cc21)N1CCOCC1",
    "NC(=O)c1cccc2nc3ccccc3nc12",
    "S=c1sscc1-c1ccccc1",
    "O=[N+]([O-])OCCOCCO[N+](=O)[O-]",
    "CC(=O)Nc1ccc2c3c(ccc([N+](=O)[O-])c13)CC2",
    "COc1c(CO)cc(Cl)cc1CO",
    "Cc1cccc[c]1[Hg][c]1ccccc1C",
    "c1ccc(-c2ccc(-c3ccc(-c4ccccc4)cc3)cc2)cc1",
    "C=CCSSSCC=C",
    "COc1cc(CO)c(CO)cc1OC",
    "Clc1cccnc1",
    "CCCC/N=N/N(C)Cc1ccccc1",
    "CCCCOC(N)=O",
    "Cc1ccccc1OOc1ccccc1C",
    "Cc1ccccc1/N=N/c1ccccc1C",
    "Cc1ccccc1P(c1ccccc1C)c1ccccc1C",
    "CCCCCCCCCCc1ccc2c(c1)N(C)C(C(C)C)C(=O)NC(CO)C2",
    "CCCCCCCCCCCCCCCCNS(=O)(=O)c1ccc(S(=O)(=O)F)cc1",
    "c1ccc(CN(Cc2ccccc2)N=NN(Cc2ccccc2)Cc2ccccc2)cc1",
    "OC(CNCCS)C(O)CNCCS",
    "CN(OC(=O)C(C)(C)C)C(=O)C(C)(C)C"
}
df = df[~df['SMILES'].isin(exclude)].reset_index(drop=True)

# STEP 5: Prepare fingerprint matrix
fp_cols = [c for c in df.columns if c.startswith('fp_')]
fps     = df[fp_cols].values.astype(bool)

# STEP 6: MaxMin diversity selection
np.random.seed(42)
n_total  = len(df)
n_pick   = 20
selected = [np.random.randint(n_total)]
min_sim  = np.ones(n_total)

for _ in range(1, n_pick):
    last = fps[selected[-1]]
    inter= np.logical_and(fps, last).sum(axis=1)
    union= np.logical_or (fps, last).sum(axis=1)
    sim  = inter / union
    min_sim = np.minimum(min_sim, sim)
    min_sim[selected] = 1.0
    selected.append(int(np.argmin(min_sim)))

# STEP 7: Gather chosen SMILES and output
chosen = df.loc[selected, 'SMILES'].tolist()

print("\n✅ 20 Newly Selected Diverse Query SMILES:\n")
for smi in chosen:
    print(smi)

out_txt = '/mnt/d/akarmark/data/diverse_20_smiles.txt'
with open(out_txt, 'w') as f:
    for smi in chosen:
        f.write(smi + "\n")

print(f"\nWritten 20 SMILES to {out_txt}")
