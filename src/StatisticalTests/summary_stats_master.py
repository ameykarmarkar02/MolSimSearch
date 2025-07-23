import pandas as pd
import numpy as np

# Paths to Excel files
chemberta_path = '/mnt/d/akarmark/Results/Data/chemBERTa.xlsx'
selformer_path = '/mnt/d/akarmark/Results/Data/SELFormer.xlsx'
ecfp4_path = '/mnt/d/akarmark/Results/Data/ECP4fingerprints.xlsx'

# Lists for summary stats
summary_ged = []
summary_ham = []

def load_transformer(path, sheet, top_n=35):
    """
    Load transformer-based sheets (chemBERTa or SELFormer).
    Detect header row containing 'Molecule ID' and parse the table.
    Only retain the first top_n hits.
    """
    df_raw = pd.read_excel(path, sheet_name=sheet, header=None)
    hdr_idx = df_raw.index[
        df_raw.apply(lambda r: r.astype(str).str.contains('Molecule ID').any(), axis=1)
    ][0]
    header = df_raw.iloc[hdr_idx].tolist()
    df = df_raw.iloc[hdr_idx+1:].copy()
    df.columns = header
    df = df[['Distance', 'GED', 'Hamming']]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['GED'])
    df = df[df['Hamming'] != -1]
    return df.head(top_n)


def load_ecfp4(path, sheet, top_n=35):
    """
    Load ECFP4 sheets where header is proper.
    Only retain the first top_n hits.
    """
    df = pd.read_excel(path, sheet_name=sheet)
    df = df[['Distance', 'GED', 'Hamming']]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['GED'])
    df = df[df['Hamming'] != -1]
    return df.head(top_n)


def summarize(df_all, method_label):
    """
    Compute summary stats for a concatenated DataFrame df_all.
    """
    summary_ged.append({
        'Method': method_label,
        'Mean GED': df_all['GED'].mean(),
        'Median GED': df_all['GED'].median(),
        'Std GED': df_all['GED'].std(ddof=0),
        'N (pairs)': len(df_all)
    })
    summary_ham.append({
        'Method': method_label,
        'Mean Hamming': df_all['Hamming'].mean(),
        'Median Hamming': df_all['Hamming'].median(),
        'Std Hamming': df_all['Hamming'].std(ddof=0),
        'N (pairs)': len(df_all)
    })


if __name__ == '__main__':
    # 1. chemBERTa (SMILES & SELFIES)
    xl_cb = pd.ExcelFile(chemberta_path)
    for method in ['SMILES', 'SELFIES']:
        sheets = [s for s in xl_cb.sheet_names if s.lower().endswith(f'_{method.lower()}')]
        df_all = pd.concat([
            load_transformer(chemberta_path, s) for s in sheets
        ], ignore_index=True)
        summarize(df_all, f'chemBERTa ({method})')

    # 2. SELFormer (SMILES & SELFIES)
    xl_sf = pd.ExcelFile(selformer_path)
    for method in ['SMILES', 'SELFIES']:
        sheets = [s for s in xl_sf.sheet_names if s.lower().endswith(f'_{method.lower()}')]
        df_all = pd.concat([
            load_transformer(selformer_path, s) for s in sheets
        ], ignore_index=True)
        summarize(df_all, f'SELFormer ({method})')

    # 3. ECFP4 (cosine & Tanimoto)
    xl_ec = pd.ExcelFile(ecfp4_path)
    df_cos = pd.concat([
        load_ecfp4(ecfp4_path, s) for s in xl_ec.sheet_names if 'cosine' in s.lower()
    ], ignore_index=True)
    summarize(df_cos, 'ECFP4 (cosine sim)')

    df_tan = pd.concat([
        load_ecfp4(ecfp4_path, s) for s in xl_ec.sheet_names if 'tanimoto' in s.lower()
    ], ignore_index=True)
    summarize(df_tan, 'ECFP4 (Tanimoto)')

    # Convert to DataFrames
    df_ged_summary = pd.DataFrame(summary_ged)
    df_ham_summary = pd.DataFrame(summary_ham)

    # Print to console
    print("Table 1. Summary of Graph Edit Distances (GED):")
    print(df_ged_summary.to_string(index=False, float_format='%.2f'))

    print("\nTable 2. Summary of Hamming Distances:")
    print(df_ham_summary.to_string(index=False, float_format='%.2f'))

    # Export to CSV
    ged_csv = 'ged_summary.csv'
    ham_csv = 'hamming_summary.csv'
    df_ged_summary.to_csv(ged_csv, index=False)
    df_ham_summary.to_csv(ham_csv, index=False)

    print(f"\nExported GED summary to '{ged_csv}' and Hamming summary to '{ham_csv}'.")
