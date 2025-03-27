import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
from streamlit.components.v1 import html

# Load data
@st.cache_data
def load_data():
    ppi_data = pd.read_csv("new_dataset.csv")
    embeddings_data = pd.read_csv("protein_embeddings.csv")
    
    # Extract embedding dimensions (Dim_0 to Dim_1023)
    embedding_cols = [col for col in embeddings_data.columns if col.startswith("Dim_")]
    embeddings = {}
    for _, row in embeddings_data.iterrows():
        embedding = row[embedding_cols].values.tolist()  # Convert dimensions to a list
        embeddings[row["Protein_ID"]] = embedding
    return ppi_data, embeddings

# Build the full PPI graph
@st.cache_resource
def build_graph(ppi_data, embeddings):
    G = nx.Graph()
    for _, row in ppi_data.iterrows():
        G.add_edge(row["protein1"], row["protein2"], 
                   weight=row["combined_score"],
                   # Store additional evidence for tooltips
                   neighborhood=row["neighborhood"],
                   coexpression=row["coexpression"],
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
    
    # Add edges with weights and score-based styling
    for edge in subgraph.edges(data=True):
        weight = edge[2]["weight"]
        # Normalize combined_score (assuming 0-1000 scale) for visualization
        normalized_weight = weight / 1000.0
        # Color edges based on combined_score
        color = "green" if weight >= 800 else "orange" if weight >= 600 else "gray"
        net.add_edge(edge[0], edge[1], 
                     value=normalized_weight * 10,  # Scale for visibility
                     color=color,
                     title=f"Combined Score: {weight}\nNeighborhood: {edge[2]['neighborhood']}\nCoexpression: {edge[2]['coexpression']}\nDatabase: {edge[2]['database']}\nTextmining: {edge[2]['textmining']}")
    
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
G = build_graph(ppi_data, embeddings)

# Initialize session state
if "selected_proteins" not in st.session_state:
    st.session_state.selected_proteins = ["9606.ENSP00000000233"]  # Start with the first protein

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
    st.session_state.selected_proteins = ["9606.ENSP00000000233"]
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