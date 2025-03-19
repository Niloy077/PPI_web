# precompute_2d.py
import pickle
import numpy as np
from sklearn.decomposition import PCA

# Load embeddings
with open("protbert_embeddings.pkl", "rb") as file:
    embeddings = pickle.load(file)

protein_ids = list(embeddings.keys())
embedding_matrix = np.array(list(embeddings.values()))

# Reduce to 2D
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embedding_matrix)

# Save as a new pickle file
with open("reduced_embeddings.pkl", "wb") as file:
    pickle.dump({"ids": protein_ids, "coords": reduced_embeddings}, file)

print("2D coordinates saved to reduced_embeddings.pkl")