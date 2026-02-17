from rag.vector_store import collection

def retrieve_rules(claim_data: dict, shap_factors: list, top_k: int = 3):
    query_text = f"""
    Claim data: {claim_data}
    Key risk factors: {shap_factors}
    """
    
    results = collection.query(
        query_texts=[query_text],
        n_results = top_k
    )
    
    return results["documents"][0]