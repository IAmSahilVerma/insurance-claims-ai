def retrieve_rules():
    with open("fraud_rules.txt", "r") as f:
        rules = f.read()
    return rules