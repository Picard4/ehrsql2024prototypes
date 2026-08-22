import json
import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import torch
import multiprocessing as mp
from scoring.scoring_utils import reliability_score, penalize, execute_all_distributed, execute_all
from scoring.postprocessing import post_process_sql

"""Score Calculation Function (taken from scoring.py)"""
def calculate_score_from_files(reference_path, prediction_path):
    with open(reference_path) as f:
        real_dict = json.load(f)
    with open(prediction_path) as f:
        pred_dict = json.load(f)
    assert set(real_dict) == set(pred_dict), "IDs do not match"

    real_dict = {id_: post_process_sql(real_dict[id_]) for id_ in real_dict}
    pred_dict = {id_: post_process_sql(pred_dict[id_]) for id_ in pred_dict}

    current_real_dir = os.path.dirname(os.path.realpath(__file__))
    db_path = os.path.join(current_real_dir, 'mimic_iv.sqlite')
    if not os.path.exists(db_path):
        raise Exception('File does not exist: %s' % db_path)

    num_workers = min(2, mp.cpu_count())
    if num_workers > 1:
        real_result = execute_all_distributed(real_dict, db_path, tag='real', num_workers=num_workers)
        pred_result = execute_all_distributed(pred_dict, db_path, tag='pred', num_workers=num_workers)
    else:
        real_result = execute_all(real_dict, db_path, tag='real')
        pred_result = execute_all(pred_dict, db_path, tag='pred')

    scores = reliability_score(real_result, pred_result)
    accuracy0 = penalize(scores, penalty=0)
    accuracy5 = penalize(scores, penalty=5)
    accuracy10 = penalize(scores, penalty=10)
    accuracyN = penalize(scores, penalty=len(scores))

    scores_dict = {
        'accuracy0': accuracy0*100,
        'accuracy5': accuracy5*100,
        'accuracy10': accuracy10*100,
        'accuracyN': accuracyN*100
    }
    return scores_dict

def generate_query(model, tokenizer, role_string, question):
    messages = [
        {"role": "system", "content": role_string},
        {"role": "user", "content": question}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    prediction = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return prediction

def calculate_score_from_model(model, role_string, dataset, tokenizer, reference_dir, reference_file_name, prediction_dir, prediction_file_name):
    results = {}
    reference_path = os.path.join(reference_dir, reference_file_name)
    prediction_path = os.path.join(prediction_dir, prediction_file_name)
    print(f"Predictions will be saved to {prediction_path} and compared to the contents of {reference_path}")
    for example in dataset['data']:
        query = generate_query(model, tokenizer, role_string, example['question'])
        results[example['id']] = query
    os.makedirs(prediction_dir, exist_ok=True)
    with open(prediction_path, 'w') as f:
        json.dump(results, f, indent=4)
    return calculate_score_from_files(reference_path, prediction_path)