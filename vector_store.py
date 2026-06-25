import numpy as np
import faiss
import pickle
from embeddings import get_embedding

class VectorStore:
    def __init__(self):
        self.index = None
        self.nodes = []

    def build(self, nodes):
        self.nodes = nodes
        embeddings = []
        for node in nodes:
            embeddings.append(
                get_embedding(node.text)
            )
        embeddings = np.array(
            embeddings,
            dtype=np.float32
        )
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        print(f"Indexed {len(nodes)} nodes.")

    def search(self, query, top_k=3):
        q = get_embedding(query)
        q = np.array([q], dtype=np.float32)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, top_k)
        return scores[0], indices[0]

    def save(self):
        faiss.write_index(
            self.index,
            "storage/faiss.index"
        )
        with open("storage/nodes.pkl", "wb") as f:
            pickle.dump(self.nodes, f)
        print("Index saved.")

    def load(self):
        self.index = faiss.read_index(
            "storage/faiss.index"
        )
        with open("storage/nodes.pkl", "rb") as f:
            self.nodes = pickle.load(f)
        print("Index loaded.")