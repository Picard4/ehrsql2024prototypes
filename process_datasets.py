import sys

import json

train_file = "./datasets-unprocessed/train_annotated.json"
val_file = "./datasets-unprocessed/val_annotated.json"
test_file = "./datasets-unprocessed/test_annotated.json"
processed_train_file = "./datasets-processed/train_processed.json"
processed_val_file = "./datasets-processed/val_processed.json"
processed_test_file = "./datasets-processed/test_processed.json"

# You only need to run this once to prep the files.
def process_json_file(json_file, processed_json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    for item in data:
        item.pop('val_dict', None)
        item.pop('id', None)
        item.pop('template', None)
    with open(processed_json_file, 'w') as file:
        json.dump(data, file, indent=4)

process_json_file(train_file, processed_train_file)
process_json_file(val_file, processed_val_file)
process_json_file(test_file, processed_test_file)