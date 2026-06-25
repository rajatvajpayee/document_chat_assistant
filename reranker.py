from sentence_transformers import CrossEncoder


model = CrossEncoder(
    "BAAI/bge-reranker-base"
)


def rerank(query, retrieved_docs):
    pairs = []
    for doc in retrieved_docs:
        pairs.append(
            (query, doc["text"])
        )
    scores = model.predict(pairs)
    for doc, score in zip(retrieved_docs, scores):
        doc["r_score"] = float(score)
    retrieved_docs.sort(
        key=lambda x: x["r_score"],
        reverse=True
    )
    return retrieved_docs