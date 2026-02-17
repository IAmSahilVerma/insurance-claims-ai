from rag.retriever import retrieve_rules
from rag.vector_store import collection

fraud_rules = retrieve_rules()

for i, rule in enumerate(fraud_rules):
    collection.add(
        documents=[rule],
        ids=[f"rule_{i}"]
    )
    
print("Fraud rules loaded into Vector DB.")