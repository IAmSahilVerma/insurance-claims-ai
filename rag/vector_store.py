import chromadb
from chromadb.utils import embedding_functions

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")

client = chromadb.Client()

collection = client.get_or_create_collection(
    name="fraud_knowledge_base",
    embedding_function=embedding_function
)