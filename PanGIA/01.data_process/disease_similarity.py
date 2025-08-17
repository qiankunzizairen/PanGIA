import pandas as pd
import numpy as np
import obonet
import networkx as nx
import math

# === Load the Disease Ontology graph ===
url = "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/doid.obo"
HDO_Net = obonet.read_obo(url)

# === Compute the semantic value vector for a given disease DOID ===
def get_SV(disease, w):
    S = HDO_Net.subgraph(nx.descendants(HDO_Net, disease) | {disease})
    SV = dict()
    shortest_paths = nx.shortest_path(S, source=disease)
    for x in shortest_paths:
        SV[x] = math.pow(w, (len(shortest_paths[x]) - 1))
    return SV

# === Compute the semantic similarity between two DOIDs ===
def get_similarity(d1, d2, w):
    SV1 = get_SV(d1, w)
    SV2 = get_SV(d2, w)
    intersection_value = 0
    for disease in set(SV1.keys()) & set(SV2.keys()):
        intersection_value += SV1[disease]
        intersection_value += SV2[disease]
    return intersection_value / (sum(SV1.values()) + sum(SV2.values()))

# === Construct the disease-disease similarity matrix ===
def get_disease_similarity_matrix(num_diseases, doid_list, w):
    similarity_matrix = np.zeros((num_diseases, num_diseases))
    for i in range(num_diseases):
        if doid_list[i] in HDO_Net.nodes:
            for j in range(i + 1, num_diseases):
                if doid_list[j] in HDO_Net.nodes:
                    sim = get_similarity(doid_list[i], doid_list[j], w)
                    similarity_matrix[i, j] = similarity_matrix[j, i] = sim
    return similarity_matrix

# === Load the disease-DOID mapping file ===
doid_df = pd.read_excel('./data/all_disease_with_doid.xlsx')  # Modify the path as needed

# === Extract disease names and corresponding DOIDs ===
disease_names = list(doid_df['disease'])
doid_list = list(doid_df['doid'])

# === Compute the similarity matrix ===
similarity_matrix = get_disease_similarity_matrix(len(doid_list), doid_list, w=0.5)
np.fill_diagonal(similarity_matrix, 1)  # Set self-similarity to 1

# === Save the result to a CSV file ===
d2d_df = pd.DataFrame(similarity_matrix, columns=disease_names, index=disease_names)
d2d_df.to_csv('./data/d2d_do.csv')  # Recommended output path inside ./data

print("Disease similarity computation completed. File saved to ./data/d2d_do.csv")