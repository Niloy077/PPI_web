import streamlit as st
from Bio import PDB
from Bio.SeqUtils import seq1
from transformers import EsmModel, EsmTokenizer
import numpy as np
import torch
import matplotlib.pyplot as plt
import io

# Load ESM-2 Model & Tokenizer
@st.cache_resource
def load_model():
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Function to extract sequence from PDB file
def extract_sequence_from_pdb(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    seq = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue, standard=True):
                    seq += seq1(residue.get_resname())
    return seq

# Default PDB files (Stored in a local folder "data/")
DEFAULT_PDB_FILES = {
    "Protein 1 (1A3N)": "data/1a3n.pdb",
    "Protein 2 (9J82)": "data/9j82.pdb",
    "Protein 3 (2DN2)": "data/2dn2.pdb",
    "Protein 4 (4QQI)": "data/4qqi.pdb"
}
# Streamlit UI
st.title("🔬 Protein Embedding Visualizer")
st.write("Upload one or more PDB files to extract sequences and compute embeddings.")

uploaded_files = st.file_uploader("Upload PDB Files", type=["pdb"], accept_multiple_files=True)

if uploaded_files:
    embeddings_list = []
    protein_names = []

    for file in uploaded_files:
        # Read PDB file from user upload
        pdb_content = file.read()
        pdb_io = io.StringIO(pdb_content.decode("utf-8"))
        
        # Extract sequence
        seq = extract_sequence_from_pdb(pdb_io)
        protein_names.append(file.name.replace(".pdb", ""))  # Extract protein name
        
        # Compute embeddings
        inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()  # Get protein-level embedding
        embeddings_list.append(embedding)

    # Convert embeddings to numpy array
    embedding_matrix = np.vstack(embeddings_list)  # Shape: (num_proteins, embedding_size)

    # Extract first two dimensions directly (assuming embeddings have at least 2 dimensions)
    embedding_2d = embedding_matrix[:, :2]  

    # Scatter plot of protein embeddings
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c='blue', s=100)

    # Annotate protein names
    for i, name in enumerate(protein_names):
        ax.text(embedding_2d[i, 0], embedding_2d[i, 1], name, fontsize=12, ha="right")

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title("Protein Embedding Comparison (Raw 2D Projection)")

    # Show plot in Streamlit
    st.pyplot(fig)
