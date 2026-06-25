from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
import retriever
from vector_store import VectorStore
from embeddings import get_embedding

from prompt import build_prompt
from llm import generate
from retriever import Retriever


documents = SimpleDirectoryReader("data").load_data()

splitter = SentenceSplitter(
    chunk_size=500,
    chunk_overlap=100
)

nodes = splitter.get_nodes_from_documents(documents)

store = VectorStore()
store.load()

# query = input("Ask something: ")
query = "What is multi-head attention"

rtrvr = Retriever(store)
retrieved_nodes = rtrvr.retrieve(query, top_k=10)

scores, indices = store.search(query)

context = ""
for node in retrieved_nodes:
    context += "\n--------------------------"
    context += node["text"] + '\n'
    context += "Confidence :"+ str(node["score"])
    context += "\n\n"

prompt = build_prompt(
    context=context,
    question=query
)    


print(prompt)
# answer = generate(prompt)
# 
# print(answer)