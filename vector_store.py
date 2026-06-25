import numpy as np
import faiss

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