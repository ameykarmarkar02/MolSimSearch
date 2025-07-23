from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Paths to Excel files
chemberta_path = '/mnt/d/akarmark/Results/Data/chemBERTa.xlsx'
selformer_path = '/mnt/d/akarmark/Results/Data/SELFormer.xlsx'
ecfp4_path = '/mnt/d/akarmark/Results/Data/ECP4fingerprints.xlsx'

def load_transformer(path, sheet, top_n=10):
    """
    Load transformer-based sheets (chemBERTa or SELFormer).
    Detect header row containing 'Molecule ID' and parse the table.
    Only retain the first top_n molecules.
    """
    df_raw = pd.read_excel(path, sheet_name=sheet, header=None, engine='openpyxl')
    matches = df_raw.index[
        df_raw.apply(lambda r: r.astype(str).str.contains('Molecule ID', na=False).any(), axis=1)
    ]
    if matches.empty:
        return pd.DataFrame(columns=['Distance', 'GED', 'Hamming'])
    hdr_idx = matches[0]
    header = df_raw.iloc[hdr_idx].tolist()
    df = df_raw.iloc[hdr_idx+1:].copy()
    df.columns = header
    # Keep only required columns
    df = df[['Distance', 'GED', 'Hamming']]
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['GED'])
    df = df[df['Hamming'] != -1]
    # Only take top_n rows
    return df.head(top_n)


def load_ecfp4(path, sheet, top_n=10):
    """
    Load ECFP4 sheets (cosine or tanimoto).
    Only retain the first top_n molecules.
    """
    df = pd.read_excel(path, sheet_name=sheet, engine='openpyxl')
    if not set(['Distance', 'GED', 'Hamming']).issubset(df.columns):
        return pd.DataFrame(columns=['Distance', 'GED', 'Hamming'])
    df = df[['Distance', 'GED', 'Hamming']]
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['GED'])
    df = df[df['Hamming'] != -1]
    # Only take top_n rows
    return df.head(top_n)

# Collect data for all methods (only top 5 per sheet)
records = []

def append_records(df, label):
    for _, row in df.iterrows():
        records.append({'Method': label, 'GED': row['GED'], 'Hamming': row['Hamming']})

# chemBERTa
for method in ['SMILES', 'SELFIES']:
    xl = pd.ExcelFile(chemberta_path, engine='openpyxl')
    sheets = [s for s in xl.sheet_names if s.lower().endswith(f'_{method.lower()}')]
    for sheet in sheets:
        df = load_transformer(chemberta_path, sheet)
        append_records(df, f'chemBERTa ({method})')

# SELFormer
for method in ['SMILES', 'SELFIES']:
    xl = pd.ExcelFile(selformer_path, engine='openpyxl')
    sheets = [s for s in xl.sheet_names if s.lower().endswith(f'_{method.lower()}')]
    for sheet in sheets:
        df = load_transformer(selformer_path, sheet)
        append_records(df, f'SELFormer ({method})')

# ECFP4
xl = pd.ExcelFile(ecfp4_path, engine='openpyxl')
cos_sheets = [s for s in xl.sheet_names if 'cosine' in s.lower()]
tan_sheets = [s for s in xl.sheet_names if 'tanimoto' in s.lower()]

for sheet in cos_sheets:
    df = load_ecfp4(ecfp4_path, sheet)
    append_records(df, 'ECFP4 (cosine sim)')

for sheet in tan_sheets:
    df = load_ecfp4(ecfp4_path, sheet)
    append_records(df, 'ECFP4 (Tanimoto)')

# Build DataFrame
data = pd.DataFrame.from_records(records)

# ONE-WAY ANOVA for GED
model_ged = ols('GED ~ C(Method)', data=data).fit()
anova_ged = sm.stats.anova_lm(model_ged, typ=2)
print("ANOVA for GED:\n", anova_ged, "\n")

# ONE-WAY ANOVA for Hamming
model_ham = ols('Hamming ~ C(Method)', data=data).fit()
anova_ham = sm.stats.anova_lm(model_ham, typ=2)
print("ANOVA for Hamming:\n", anova_ham, "\n")

# Post-hoc Tukey HSD for GED
tukey_ged = pairwise_tukeyhsd(endog=data['GED'], groups=data['Method'], alpha=0.05)
print("Tukey HSD for GED:\n", tukey_ged, "\n")

# Post-hoc Tukey HSD for Hamming
tukey_ham = pairwise_tukeyhsd(endog=data['Hamming'], groups=data['Method'], alpha=0.05)
print("Tukey HSD for Hamming:\n", tukey_ham)
