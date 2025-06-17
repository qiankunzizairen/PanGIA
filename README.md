# PanGIA: A universal framework for identifying association between ncRNAs and diseases

PanGIA is a deep learning model for predicting ncRNA-disease associations.
## Model Architecture
![](PanGIA/architecture.png)

## Installation
```bash
conda create -n pangia python=3.11
conda activate pangia
pip install -r requirements.txt
```

## Prepare Datasets
The raw data can be downloaded from the following sources:
- **miRNA**: The associations between miRNAs and diseases were obtained from the HMDD v4.0 database, while the sequence information of miRNAs was retrieved from the miRBase database.

- **LncRNA/circRNA**: This study includes lncRNA and circRNA associations with diseases, with data obtained from LncRNADisease v3.0. The sequence information of circRNAs was retrieved from the circBase database. In contrast, lncRNA sequences were collected from two sources: GENCODE and NONCODE.

- **piRNA**: The associations between piRNAs and diseases were obtained from the piRDisease v1.0 database, and the sequence information was retrieved from the piRBase and piRNAdb databases.

- **Disease**: This study utilizes Disease Ontology Identifiers (DOIDs) to construct the disease similarity matrix, with corresponding information obtained from the Disease Ontology database.

These data are also organized in the ./data folder.

## Quick Start
### 1.Data Preprocessing & Cleaning
Prepare the RNA sequence files for each RNA type (`miRNA`, `piRNA`, `lncRNA`, `circRNA`) in CSV format:

```bash
# Example format (no header):
# RNA_ID,Sequence
miR0001,AGCUUGGA...
miR0002,CGAUUAGC...
```
Run the script to perform global alignment of RNA sequences and compute their pairwise similarity:
```bash
python compute_RNA_similarity.py
```
Next, merge the RNA sequence similarity matrices across all RNA types (miRNA, piRNA, lncRNA, circRNA) into a unified format for downstream analysis:

```bash
python merge_RNA_similarity_matrices.py
```
This script reads the normalized pairwise similarity matrices generated for each RNA type and combines them into a multi-view or unified similarity representation for further modeling.

Next, run the script `compute_disease_similarity.py` to generate the disease ontology-based similarity matrix:

```bash
python compute_disease_similarity.py
```

This script calculates pairwise semantic similarities between diseases based on the Disease Ontology (DO) structure, and saves the resulting matrix to:

```
./data/d2d_do.csv
```
Run the following script to generate the binary association matrix between ncRNAs and diseases:

```bash
python generate_RD_adj.py
```

This script constructs the ncRNA–disease adjacency matrix based on known associations.  
The output is a matrix where each row represents an ncRNA and each column represents a disease,  
with entries marked as 1 if an association exists, and 0 otherwise.

Next, pretrain Word2Vec embeddings for RNA k-mer segments using the following script:

```bash
python pretrain_RNA_kmer.py
```

This script tokenizes RNA sequences into k-mers, performs sliding-window segmentation, pads them to a unified length, and trains Word2Vec embeddings for each RNA type (miRNA, circRNA, lncRNA, piRNA).  
The output includes:

- `gensim_feat_<type>_<VECTOR_SIZE>.npy`: A dictionary containing
  - k-mer embedding matrix
  - padded k-mer ID sequences
  - segment-to-sequence mapping
  
Run the programs in the `build dataset` folder sequentially to generate the cross-validation dataset.

### 2.Model Training
Use the processed similarity matrices and datasets to train the model:
```bash
python main.py
```
Due to the high memory/GPU usage of the network, please pass parameters when running main.py to control the network size according to your own computational resources.

**Note:** Due to the large size of the dataset, which exceeds GitHub's upload limit, the data are provided via an external link:  
[https://drive.google.com/drive/folders/1KhwwvnSsRZNpuDv1SJjNkrTRHXQ2EUsM?usp=drive_link](https://drive.google.com/drive/folders/1KhwwvnSsRZNpuDv1SJjNkrTRHXQ2EUsM?usp=drive_link)
