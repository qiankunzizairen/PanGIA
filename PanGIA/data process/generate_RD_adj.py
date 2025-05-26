import pandas as pd

# === Load input files ===
miRDA_df = pd.read_excel('./data/miRDA.xlsx')          # miRNA-disease associations
LncCirDA_df = pd.read_excel('./data/LncCirDA.xlsx')    # lncRNA/circRNA-disease associations
piRDA_df = pd.read_excel('./data/piRDA.xlsx')          # piRNA-disease associations
allR_id_df = pd.read_excel('./data/allR_id.xlsx')      # RNA ID mapping
disease_id_df = pd.read_excel('./data/disease_ID.xlsx')  # Disease ID mapping

# === Map miRNA and disease names to IDs ===
miRDA_df['RNA_ID'] = miRDA_df['miRNA'].map(allR_id_df.set_index('RNA')['ID'].to_dict())
miRDA_df['Disease_ID'] = miRDA_df['disease'].map(disease_id_df.set_index('disease')['ID'].to_dict())

# === Map lncRNA/circRNA and diseases to IDs ===
LncCirDA_df['RNA_ID'] = LncCirDA_df['ncRNA Symbol'].map(allR_id_df.set_index('RNA')['ID'].to_dict())
LncCirDA_df['Disease_ID'] = LncCirDA_df['Disease Name'].map(disease_id_df.set_index('disease')['ID'].to_dict())

# === Map piRNA and diseases to IDs ===
piRDA_df['RNA_ID'] = piRDA_df['RNA Symbol'].map(allR_id_df.set_index('RNA')['ID'].to_dict())
piRDA_df['Disease_ID'] = piRDA_df['Disease Name'].map(disease_id_df.set_index('disease')['ID'].to_dict())

# === Prepare mapping dictionaries ===
rna_names = allR_id_df.set_index('ID')['RNA'].to_dict()
disease_names = disease_id_df.set_index('ID')['disease'].to_dict()
rna_ids = allR_id_df['ID'].tolist()
disease_ids = disease_id_df['ID'].tolist()

# === Initialize a zero-filled adjacency matrix ===
adj_matrix = pd.DataFrame(0, index=rna_ids, columns=disease_ids)

# === Helper function to fill the matrix from a dataframe ===
def fill_matrix(df, matrix):
    for _, row in df.iterrows():
        if pd.notna(row['RNA_ID']) and pd.notna(row['Disease_ID']):
            matrix.loc[row['RNA_ID'], row['Disease_ID']] = 1.0
    return matrix

# === Fill the matrix using all three association sources ===
adj_matrix = fill_matrix(miRDA_df, adj_matrix)
adj_matrix = fill_matrix(LncCirDA_df, adj_matrix)
adj_matrix = fill_matrix(piRDA_df, adj_matrix)

# === Rename rows and columns using actual RNA and disease names ===
adj_matrix.index = adj_matrix.index.map(rna_names)
adj_matrix.columns = adj_matrix.columns.map(disease_names)

# === Save the final matrix ===
adj_matrix.to_csv('./data/RDA_Matrix.csv')

print("RNA–Disease adjacency matrix has been successfully generated and saved as './data/RDA_Matrix.csv'.")