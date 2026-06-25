from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
import retriever
from vector_store import VectorStore
from embeddings import get_embedding

from prompt import build_prompt
from llm import generate
from retriever import Retriever

from reranker import rerank


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

#now rerank 
retrieved_nodes= rerank(query, retrieved_nodes)
top_n = 5 # lets select top 5 documents now.

context = ""
for node in retrieved_nodes[:top_n]:
    context += "--------------------------\n"
    context += node["text"] + '\n'
    context += "Before mConfidence :"+ str(node["score"]) + "After Confidence :" + str(node["r_score"])+ "\n"
    context += "\n\n"

prompt = build_prompt(
    context=context,
    question=query
)    


print(prompt)
# answer = generate(prompt)
# 
# print(answer)