import json
from itertools import islice

if __name__ == "__main__":
    file_path = "./data/sr.json"
    with open(file_path, 'r') as file:
        data = json.load(file)
    print(len(data))

    key_value_pairs = list(islice(data.items(), 1603, 1603 + 1))
    print(key_value_pairs)