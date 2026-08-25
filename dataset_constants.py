import os

processed_train_file = "./datasets-processed/train_processed.json"
processed_val_file = "./datasets-processed/val_processed.json"

# File path constants
data_file_name = 'data.json'
reference_file_name = 'label.json'
prediction_file_name = 'prediction.json'
data_dir = os.path.join('./inference/', 'data')
reference_dir = os.path.join('./inference/', 'ref')
prediction_dir = os.path.join('./inference/training', 'res')

# Dataset access constants
input_name = "question"
label_name = "query"