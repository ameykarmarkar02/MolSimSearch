import pandas as pd
import matplotlib.pyplot as plt

# File path and hit values
file_path = '/mnt/d/akarmark/Results/Data/Graph/summary_stats_GED.xlsx'
hit_values = [3, 5, 8, 10, 15, 20, 25, 35]

# Initialize a DataFrame to collect mean Hamming distances
methods = [
    "chemBERTa (SMILES)",
    "chemBERTa (SELFIES)",
    "SELFormer (SMILES)",
    "SELFormer (SELFIES)",
    "ECFP4 (cosine sim)",
    "ECFP4 (Tanimoto)"
]
mean_hamming_df = pd.DataFrame(index=hit_values, columns=methods)

# Read each sheet and extract the mean Hamming values
for n in hit_values:
    sheet_name = f'n={n}'
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    for method in methods:
        mean_value = df.loc[df['Method'] == method, 'Mean GED'].values[0]
        mean_hamming_df.at[n, method] = mean_value

# Plotting
plt.figure(figsize=(10, 6))
for method in methods:
    plt.plot(
        mean_hamming_df.index,
        mean_hamming_df[method].astype(float),
        marker='o',
        label=method
    )

plt.xlabel('Number of Hits')
plt.ylabel('Mean GED')
plt.title('Mean GED vs Number of Hits for Different Methods')
plt.xticks(hit_values)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save to file instead of showing interactively:
output_path = '/mnt/d/akarmark/Results/Data/Graph/mean_GED_vs_hits.png'
plt.savefig(output_path, dpi=300)
plt.close()
