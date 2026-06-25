from vector_store import VectorStore
from prompt import build_prompt
from llm import generate

from retriever import Retriever
from reranker import rerank
from memory import ConversationalMemory

store = VectorStore()
store.load()

history = ConversationalMemory()

while True:

    query = input("\nAsk> ")

    if query.lower() == "exit":
        break

    rtrvr = Retriever(store)
    retrieved_nodes = rtrvr.retrieve(query, top_k=10)
    retrieved_nodes= rerank(query, retrieved_nodes)
    top_n = 5 # lets select top 5 documents now.

    mem = history.get_history()

    context = ""

    for node in retrieved_nodes:
        context += node['text'] + "\n\n"

    prompt = build_prompt(mem, context, query)

    answer = generate(prompt)
    print("\nAnswer:")
    print(answer)

    history.add_user(query)
    history.add_assistant(answer)
