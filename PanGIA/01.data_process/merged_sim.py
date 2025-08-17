import numpy as np
import pandas as pd

def load_similarity_with_ids(path):
    df = pd.read_csv(path, index_col=0)
    ids = df.index.to_numpy()
    matrix = df.to_numpy(dtype=float)
    return ids, matrix

ids1, m1 = load_similarity_with_ids("Similarity/miR_similarity.csv")
ids2, m2 = load_similarity_with_ids("Similarity/lncR_similarity.csv")
ids3, m3 = load_similarity_with_ids("Similarity/circR_similarity.csv")
ids4, m4 = load_similarity_with_ids("Similarity/piR_similarity.csv")

all_ids = np.concatenate([ids1, ids2, ids3, ids4])
sizes = [len(ids1), len(ids2), len(ids3), len(ids4)]
total_size = sum(sizes)
big_matrix = np.zeros((total_size, total_size))

start = 0
for mat in [m1, m2, m3, m4]:
    end = start + mat.shape[0]
    big_matrix[start:end, start:end] = mat
    start = end

df_out = pd.DataFrame(big_matrix, index=all_ids, columns=all_ids)
df_out.to_csv("allR_cosine_similarity.csv", float_format="%.6f")