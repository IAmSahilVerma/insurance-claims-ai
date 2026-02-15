import os

def retrieve_rules():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rules_path = os.path.join(current_dir, "fraud_rules.txt")
    with open(rules_path, "r") as f:
        rules = f.read()
    return rules

# if __name__=="__main__":
#     print(retrieve_rules())