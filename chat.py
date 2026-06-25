from vector_store import VectorStore
from prompt import build_prompt
from llm import generate

store = VectorStore()
store.load()

while True:

    query = input("\nAsk> ")

    if query.lower() == "exit":
        break

    scores, indices = store.search(query)

    context = ""

    print("\nRetrieved:\n")

    for score, idx in zip(scores, indices):

        node = store.nodes[idx]

        print(
            f"Score: {score:.3f} "
            f"(Page {node.metadata.get('page_label', '?')})"
        )

        context += node.text + "\n\n"

    prompt = build_prompt(context, query)

    answer = generate(prompt)

    print("\nAnswer\n")
    print(answer)