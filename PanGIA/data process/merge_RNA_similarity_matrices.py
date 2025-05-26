# This script merges RNA sequence similarity matrices and generates an ID mapping table for RNAs and diseases.
import pandas as pd
import numpy as np
import os

# ========== File Paths ==========
file_list = [
    './data/miR_needleman.csv',
    './data/circR_needleman.csv',
    './data/lncR_needleman.csv',
    './data/piR_needleman.csv'
]

# ========== Initialize ==========
matrices = []
rna_names = []

# ========== Read and validate each similarity matrix ==========
for file_path in file_list:
    df = pd.read_csv(file_path, index_col=0)  # Assume first column is index, first row is header
    if not np.allclose(df.values, df.values.T, atol=1e-8):
        raise ValueError(f"The matrix in {file_path} is not symmetric.")

    matrices.append(df)
    rna_names.extend(df.index.tolist())

# ========== Construct block-diagonal matrix ==========
total_size = sum(m.shape[0] for m in matrices)
final_matrix = np.zeros((total_size, total_size))

start = 0
for mat in matrices:
    size = mat.shape[0]
    final_matrix[start:start + size, start:start + size] = mat.values
    start += size

# ========== Create final DataFrame with row/column names ==========
full_index = rna_names
final_df = pd.DataFrame(final_matrix, index=full_index, columns=full_index)

# ========== Save the merged similarity matrix ==========
final_df.to_csv('./data/allR_needleman.csv', encoding='utf-8-sig')

# ========== Create and save ID-to-RNA mapping table ==========
id_df = pd.DataFrame({
    'ID': range(1, len(rna_names) + 1),
    'RNA': rna_names
})
id_df.to_excel('./data/allR_id.xlsx', index=False)

print(" Processing completed. Output files have been saved:")
print("  - ./data/allR_needleman.csv")
print("  - ./data/allR_id.xlsx")