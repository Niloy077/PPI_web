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

# Default PDB files (Stored online)
DEFAULT_PDB_FILES = {
    "Protein 1 (1A3N)": "https://files.rcsb.org/download/1A3N.pdb",
    "Protein 2 (4QQI)": "https://files.rcsb.org/download/4QQI.pdb",
    "Protein 3 (2DN2)": "https://files.rcsb.org/download/2DN2.pdb",
    "Protein 4 (9J82)": "https://files.rcsb.org/download/9J82.pdb"
}

# Fetch PDB file from URL
def fetch_pdb_from_url(url, name):
    response = requests.get(url)
    if response.status_code == 200:
        pdb_file = io.StringIO(response.text)  # Convert to file-like object
        pdb_file.name = name + ".pdb"  # Manually set a name attribute
        return pdb_file
    else:
        st.error(f"Failed to fetch PDB from {url}")
        return None

# UI: Logo and Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("logo.png", width=150)  # Adjust width as needed

st.title("🔬 Protein Embedding Visualizer")
st.subheader("Analyze and compare protein structures efficiently.")
st.write("Upload your own **PDB files**, or select from our default dataset.")

# User File Upload or Default Selection
uploaded_files = st.file_uploader("📂 Upload PDB Files", type=["pdb"], accept_multiple_files=True)
use_default = st.checkbox("Use default PDB files")

if use_default:
    selected_pdbs = st.multiselect("Select default proteins:", list(DEFAULT_PDB_FILES.keys()))  
    uploaded_files = uploaded_files or []  

    for pdb_name in selected_pdbs:
        pdb_file = fetch_pdb_from_url(DEFAULT_PDB_FILES[pdb_name], pdb_name)
        if pdb_file:
            uploaded_files.append(pdb_file)  # Append fetched PDB file

# Process selected/uploaded files
if uploaded_files:
    embeddings_list = []
    protein_names = []

    for file in uploaded_files:
        # Read PDB file from user upload or default file
        if isinstance(file, str):  # Default PDB (path as string)
            pdb_file_path = file
        else:  # Uploaded or fetched PDB file
            pdb_content = file.read()
            
            # Ensure we handle both uploaded and default files correctly
            if isinstance(pdb_content, bytes):  # If binary, decode it
                pdb_content = pdb_content.decode("latin-1")

            pdb_io = io.StringIO(pdb_content)  # Create a StringIO object
            pdb_file_path = pdb_io  # Use this for parsing

        # Extract sequence
        seq = extract_sequence_from_pdb(pdb_file_path)
        
        # Extract protein name
        if hasattr(file, "name"):  # If name exists (uploaded or default)
            protein_name = file.name.replace(".pdb", "")
        else:
            protein_name = "Unknown Protein"  # Fallback
        
        protein_names.append(protein_name)

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
