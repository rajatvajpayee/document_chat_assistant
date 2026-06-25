from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from vector_store import VectorStore
from embeddings import get_embedding

from prompt import build_prompt
from llm import generate


documents = SimpleDirectoryReader("data").load_data()

splitter = SentenceSplitter(
    chunk_size=500,
    chunk_overlap=100
)

nodes = splitter.get_nodes_from_documents(documents)

store = VectorStore()
store.build(nodes)

query = input("Ask something: ")

scores, indices = store.search(query)

context = ""
for score, idx in zip(scores, indices):
    context += nodes[idx].text
    context += "\n\n"



prompt = build_prompt(
    context=context,
    question=query
)    



answer = generate(prompt)

print(answer)