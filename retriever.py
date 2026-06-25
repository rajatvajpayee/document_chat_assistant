class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query, top_k=3):
        scores, indices = self.vector_store.search(
            query,
            top_k
        )
        results = []
        for score, idx in zip(scores, indices):
            node = self.vector_store.nodes[idx]
            results.append({
                "text": node.text,
                "score": float(score),
                "page": node.metadata.get("page_label"),
                "file": node.metadata.get("file_name")
            })

        return results