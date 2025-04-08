import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
from streamlit.components.v1 import html
import pickle

# Load data
@st.cache_data
def load_data():
    try:
        ppi_data = pd.read_csv("new_dataset.csv")
        # Load embeddings from Pickle file
        with open("protbert_embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
        return ppi_data, embeddings
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Please ensure 'new_dataset.csv' and 'protbert_embeddings.pkl' are in the 'real_time_graph_growth' folder.")
        return None, None

# Build the full PPI graph
@st.cache_resource
def build_graph(ppi_data, embeddings):
    G = nx.Graph()
    for _, row in ppi_data.iterrows():
        G.add_edge(row["protein1"], row["protein2"], 
                   weight=row["combined_score"],
                   confidence_level=row["confidence_level"],
                   confidence_level_encoded=row["confidence_level_encoded"],
                   # Store available evidence for tooltips
                   fusion=row["fusion"],
                   cooccurence=row["cooccurence"],
                   homology=row["homology"],
                   coexpression=row["coexpression"],
                   experiments=row["experiments"],
                   database=row["database"],
                   textmining=row["textmining"])
    for protein in G.nodes():
        if protein in embeddings:
            G.nodes[protein]["embedding"] = embeddings[protein]
    return G

# Create a subgraph based on selected proteins
def create_subgraph(graph, selected_proteins):
    nodes = set(selected_proteins)
    for protein in selected_proteins:
        neighbors = list(graph.neighbors(protein))
        nodes.update(neighbors)
    return graph.subgraph(nodes)

# Visualize the graph with Pyvis
def visualize_graph(subgraph, filename="graph.html"):
    net = Network(height="750px", width="100%", notebook=False)
    
    # Add nodes with embedding info in tooltip
    for node in subgraph.nodes():
        emb = subgraph.nodes[node].get("embedding", "Not available")
        # Show only the first few dimensions to avoid clutter
        emb_display = emb[:5] if isinstance(emb, list) else emb
        net.add_node(node, label=node, title=f"Protein: {node}\nEmbedding (first 5 dims): {emb_display}")
    
    # Add edges with weights and confidence-based styling
    for edge in subgraph.edges(data=True):
        weight = edge[2]["weight"]
        confidence_level = edge[2]["confidence_level"]
        confidence_level_encoded = edge[2]["confidence_level_encoded"]
        # Color edges based on confidence_level_encoded (1, 2, 3)
        color = "green" if confidence_level_encoded == 3 else "orange" if confidence_level_encoded == 2 else "gray"
        # Scale edge thickness based on combined_score (assuming 0-1 scale)
        scaled_weight = weight * 50  # Increased scaling factor for visibility
        net.add_edge(edge[0], edge[1], 
                     value=scaled_weight,  # Use for edge thickness
                     color=color,
                     title=f"Combined Score: {weight}\nConfidence Level: {confidence_level}\nConfidence Encoded: {confidence_level_encoded}\nFusion: {edge[2]['fusion']}\nCooccurence: {edge[2]['cooccurence']}\nHomology: {edge[2]['homology']}\nCoexpression: {edge[2]['coexpression']}\nExperiments: {edge[2]['experiments']}\nDatabase: {edge[2]['database']}\nTextmining: {edge[2]['textmining']}")
    
    # Physics for layout
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -3000,
          "centralGravity": 0.3,
          "springLength": 100
        }
      },
      "nodes": {
        "shape": "dot",
        "size": 20
      }
    }
    """)
    
    net.save_graph(filename)
    return filename

# Streamlit app
st.title("Real-Time PPI Graph Explorer")

# Load data and graph
ppi_data, embeddings = load_data()

# Check if data loaded successfully
if ppi_data is None or embeddings is None:
    st.stop()

G = build_graph(ppi_data, embeddings)

# Initialize session state
if "selected_proteins" not in st.session_state:
    # Use the first protein from ppi_data as the starting point
    st.session_state.selected_proteins = [ppi_data["protein1"].iloc[0]]

# Display current selected proteins
st.write("Currently exploring:", ", ".join(st.session_state.selected_proteins))

# Dropdown to add a protein
available_proteins = [p for p in G.nodes() if p not in st.session_state.selected_proteins]
new_protein = st.selectbox("Add a protein to explore:", ["None"] + available_proteins)

if new_protein != "None":
    if st.button("Add Protein"):
        st.session_state.selected_proteins.append(new_protein)
        st.rerun()

# Reset button
if st.button("Reset"):
    st.session_state.selected_proteins = [ppi_data["protein1"].iloc[0]]
    st.rerun()

# Generate and display the subgraph
subgraph = create_subgraph(G, st.session_state.selected_proteins)
html_file = visualize_graph(subgraph)

# Render the graph
with open(html_file, "r", encoding="utf-8") as f:
    html_content = f.read()
html(html_content, height=800)

# Show embeddings for selected proteins
st.subheader("Embeddings of Selected Proteins (First 5 Dimensions)")
for protein in st.session_state.selected_proteins:
    emb = G.nodes[protein].get("embedding", "Not available")
    emb_display = emb[:5] if isinstance(emb, list) else emb
    st.write(f"{protein}: {emb_display}")