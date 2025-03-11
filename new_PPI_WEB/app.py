import streamlit as st
from Bio import PDB
from Bio.SeqUtils import seq1
from transformers import EsmModel, EsmTokenizer
import numpy as np
import torch
import matplotlib.pyplot as plt
import io
import os
import requests

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

# Default PDB files (Stored as URLs)
DEFAULT_PDB_FILES = {
    "Protein 1 (1A3N)": "https://files.rcsb.org/download/1A3N.pdb",
    "Protein 2 (4QQI)": "https://files.rcsb.org/download/4QQI.pdb",
    "Protein 3 (2DN2)": "https://files.rcsb.org/download/2DN2.pdb",
    "Protein 4 (9J82)": "https://files.rcsb.org/download/9J82.pdb"
}

# Function to fetch PDB from URL
def fetch_pdb_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return io.StringIO(response.text)  # Convert to file-like object
    else:
        st.error(f"Failed to fetch PDB from {url}")
        return None

# UI: Title and Upload Section
st.title("🔬 Protein Embedding Visualizer")
st.subheader("Visualize and compare protein embeddings. 🚀")
st.write("Upload your own **PDB files**, or select from our default proteins.")

# User File Upload or Default Selection
uploaded_files = st.file_uploader("📂 Upload PDB Files", type=["pdb"], accept_multiple_files=True)
use_default = st.checkbox("Use default PDB files")

if use_default:
    selected_pdbs = st.multiselect("Select default proteins:", list(DEFAULT_PDB_FILES.keys()))  
    if selected_pdbs:
        fetched_files = [fetch_pdb_from_url(DEFAULT_PDB_FILES[pdb]) for pdb in selected_pdbs]
        uploaded_files = uploaded_files or []  # Ensure list is initialized
        uploaded_files.extend([f for f in fetched_files if f])  # Add only successful fetches

if uploaded_files:
    embeddings_list = []
    protein_names = []

    for file in uploaded_files:
        # Read PDB file from user upload or default file
        if isinstance(file, io.StringIO):  # Default PDB file fetched from URL
            pdb_io = file
            protein_names.append("Default Protein")  # Assign a generic name
        else:  # Uploaded PDB file
            pdb_content = file.read()
            if isinstance(pdb_content, bytes):  # If binary, decode it
                pdb_content = pdb_content.decode("latin-1")
            pdb_io = io.StringIO(pdb_content)
            protein_names.append(file.name.replace(".pdb", ""))  # Extract protein name

        # Extract sequence
        seq = extract_sequence_from_pdb(pdb_io)

        # Compute embeddings
        inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            outputs = model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()  # Get protein-level embedding
        embeddings_list.append(embedding)

    # Convert embeddings to numpy array
    embedding_matrix = np.vstack(embeddings_list)

    # Handle visualization for both single and multiple proteins
    if embedding_matrix.shape[0] == 1:  
        embedding_2d = np.hstack((embedding_matrix, np.zeros((1, 1))))  # Add a second dummy dimension for single protein
    else:
        embedding_2d = embedding_matrix[:, :2]

    # Scatter plot
    fig, ax = plt.subplots()
    ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c='blue', s=100)

    # Annotate protein names
    for i, name in enumerate(protein_names):
        ax.text(embedding_2d[i, 0], embedding_2d[i, 1], name, fontsize=12, ha="right")

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title("Protein Embedding Comparison (Raw 2D Projection)")

    # Show plot in Streamlit
    st.pyplot(fig)

