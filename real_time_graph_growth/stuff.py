import pickle

# Load the .pkl file
with open("protbert_embeddings.pkl", "rb") as file:
    embeddings = pickle.load(file)

# Check what’s inside
# print(type(embeddings))  # See the data type (e.g., dict, list, numpy array)
# print(embeddings)

print(len(embeddings))

protein_id = '9606.ENSP00000000233'
embedding = embeddings[protein_id]
print(embedding.shape)  # Dimensions of the vector (e.g., (1024,))
print(embedding[:10])

print(list(embeddings.keys()))