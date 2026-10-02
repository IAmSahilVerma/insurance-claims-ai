import os

import chromadb
from chromadb.utils import embedding_functions

RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fraud_rules.txt")

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# In-memory vector store. The knowledge base is small, so the rules are
# embedded every time the app starts instead of being saved to disk.
client = chromadb.Client()
collection = client.get_or_create_collection(
    name="fraud_knowledge_base",
    embedding_function=embedding_function,
)


def load_rules() -> int:
    """Load one rule per line from fraud_rules.txt into the collection."""
    with open(RULES_PATH, encoding="utf-8") as f:
        rules = [line.strip() for line in f if line.strip()]
    if rules:
        collection.upsert(documents=rules, ids=[f"rule_{i}" for i in range(len(rules))])
    return len(rules)


if collection.count() == 0:
    load_rules()
