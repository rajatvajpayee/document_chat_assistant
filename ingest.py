from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from vector_store import VectorStore

documents = SimpleDirectoryReader("data").load_data()

splitter = SentenceSplitter(
    chunk_size=500,
    chunk_overlap=100
)

nodes = splitter.get_nodes_from_documents(documents)

store = VectorStore()

store.build(nodes)

store.save()