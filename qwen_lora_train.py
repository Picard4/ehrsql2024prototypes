# import sys
# !{sys.executable} -m pip install "torch<2.5.0" "transformers>=4.42.0" huggingface_hub tiktoken einops transformers_stream_generator datasets accelerate peft ipywidgets trl

import torch
import copy
import gc
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from datasets import load_dataset, Features, Value
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from inference_utils import calculate_score_from_model

"""File Constants"""
processed_train_file = "./datasets-processed/train_processed.json"
processed_val_file = "./datasets-processed/val_processed.json"

# Inference constants
data_file_name = 'data.json'
reference_file_name = 'label.json'
prediction_file_name = 'prediction.json'
data_dir = os.path.join('./inference/', 'data')
reference_dir = os.path.join('./inference/', 'ref')
prediction_dir = os.path.join('./inference/training', 'res')

"""Functionality Constants"""
baseline_role_string = """You are a Text-to-SQL assistant for the medical database MIMIC-IV, which is designed for PostgreSQL. Output a valid SQL query for the given question or return \"null\" if it is unanswerable."""
schema_role_string = """You are a Text-to-SQL assistant for the medical database MIMIC-IV, which is designed for PostgreSQL. Output a valid SQL query for the given question or return \"null\" if it is unanswerable. The following is the database schema:
```sql
CREATE TABLE patients 
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL UNIQUE,
    gender VARCHAR(5) NOT NULL,
    dob TIMESTAMP(0) NOT NULL,
    dod TIMESTAMP(0)
);
CREATE TABLE admissions
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL UNIQUE,
    admittime TIMESTAMP(0) NOT NULL,
    dischtime TIMESTAMP(0),
    admission_type VARCHAR(50) NOT NULL,
    admission_location VARCHAR(50) NOT NULL,
    discharge_location VARCHAR(50),
    insurance VARCHAR(255) NOT NULL,
    language VARCHAR(10),
    marital_status VARCHAR(50),
    age INT NOT NULL,
    FOREIGN KEY(subject_id) REFERENCES patients(subject_id)
);
CREATE TABLE d_icd_diagnoses
(
    row_id INT NOT NULL PRIMARY KEY,
    icd_code VARCHAR(10) NOT NULL UNIQUE,
    long_title VARCHAR(255) NOT NULL
);
CREATE TABLE d_icd_procedures 
(
    row_id INT NOT NULL PRIMARY KEY,
    icd_code VARCHAR(10) NOT NULL UNIQUE,
    long_title VARCHAR(255) NOT NULL
);
CREATE TABLE d_labitems 
(
    row_id INT NOT NULL PRIMARY KEY,
    itemid INT NOT NULL UNIQUE,
    label VARCHAR(200)
);
CREATE TABLE d_items 
(
    row_id INT NOT NULL PRIMARY KEY,
    itemid INT NOT NULL UNIQUE,
    label VARCHAR(200) NOT NULL,
    abbreviation VARCHAR(200) NOT NULL,
    linksto VARCHAR(50) NOT NULL
);
CREATE TABLE diagnoses_icd
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    icd_code VARCHAR(10) NOT NULL,
    charttime TIMESTAMP(0) NOT NULL,
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id),
    FOREIGN KEY(icd_code) REFERENCES d_icd_diagnoses(icd_code)
);
CREATE TABLE procedures_icd
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    icd_code VARCHAR(10) NOT NULL,
    charttime TIMESTAMP(0) NOT NULL,
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id),
    FOREIGN KEY(icd_code) REFERENCES d_icd_procedures(icd_code)
);
CREATE TABLE labevents
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    itemid INT NOT NULL,
    charttime TIMESTAMP(0),
    valuenum DOUBLE PRECISION,
    valueuom VARCHAR(20),
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id),
    FOREIGN KEY(itemid) REFERENCES d_labitems(itemid)
);
CREATE TABLE prescriptions
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    starttime TIMESTAMP(0) NOT NULL,
    stoptime TIMESTAMP(0),
    drug VARCHAR(255) NOT NULL,
    dose_val_rx VARCHAR(100) NOT NULL,
    dose_unit_rx VARCHAR(50) NOT NULL,
    route VARCHAR(50) NOT NULL,
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id)
);
CREATE TABLE cost
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    event_type VARCHAR(20) NOT NULL,
    event_id INT NOT NULL,
    chargetime TIMESTAMP(0) NOT NULL,
    cost DOUBLE PRECISION NOT NULL,
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id),
    FOREIGN KEY(event_id) REFERENCES diagnoses_icd(row_id),
    FOREIGN KEY(event_id) REFERENCES procedures_icd(row_id),
    FOREIGN KEY(event_id) REFERENCES labevents(row_id),
    FOREIGN KEY(event_id) REFERENCES prescriptions(row_id)  
);
CREATE TABLE icustays
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    stay_id INT NOT NULL UNIQUE,
    first_careunit VARCHAR(20) NOT NULL,
    last_careunit VARCHAR(20) NOT NULL,
    intime TIMESTAMP(0) NOT NULL,
    outtime TIMESTAMP(0),
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id)
);
CREATE TABLE chartevents
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    stay_id INT NOT NULL,
    itemid INT NOT NULL,
    charttime TIMESTAMP(0) NOT NULL,
    valuenum DOUBLE PRECISION,
    valueuom VARCHAR(50),
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id),
    FOREIGN KEY(stay_id) REFERENCES icustays(stay_id),
    FOREIGN KEY(itemid) REFERENCES d_items(itemid)
);
CREATE TABLE inputevents
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    stay_id INT NOT NULL,
    starttime TIMESTAMP(0) NOT NULL,
    itemid INT NOT NULL,
    totalamount DOUBLE PRECISION,
    totalamountuom VARCHAR(50),
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id),
    FOREIGN KEY(stay_id) REFERENCES icustays(stay_id),
    FOREIGN KEY(itemid) REFERENCES d_items(itemid)
);
CREATE TABLE outputevents
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    stay_id INT NOT NULL,
    charttime TIMESTAMP(0) NOT NULL,
    itemid INT NOT NULL,
    value DOUBLE PRECISION,
    valueuom VARCHAR(50),
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id),
    FOREIGN KEY(stay_id) REFERENCES icustays(stay_id),
    FOREIGN KEY(itemid) REFERENCES d_items(itemid)
);
CREATE TABLE microbiologyevents
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    charttime TIMESTAMP(0) NOT NULL,
    spec_type_desc VARCHAR(100),
    test_name VARCHAR(100),
    org_name VARCHAR(100),
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id)
);
CREATE TABLE transfers
(
    row_id INT NOT NULL PRIMARY KEY,
    subject_id INT NOT NULL,
    hadm_id INT NOT NULL,
    transfer_id INT NOT NULL,
    eventtype VARCHAR(20) NOT NULL,
    careunit VARCHAR(20),
    intime TIMESTAMP(0) NOT NULL,
    outtime TIMESTAMP(0),
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id)
);
```"""

"""Model and Tokenizer Instantiation (and Role String + Jinja modifications)"""
model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
baseline_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# pad_token setttings to prevent logs from getting flooded
tokenizer.pad_token = tokenizer.eos_token
baseline_model.generation_config.pad_token_id = tokenizer.pad_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id

# Necessary formatting to make Qwen understand the messages and calculate the loss only on the SQL
tokenizer_jinja_old = """{%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"""

tokenizer_jinja_new = """{%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" and not message.tool_calls %}
        {{- '<|im_start|>assistant\\n' }}
        {%- generation %}
        {{- message.content }}
        {%- endgeneration %}
        {{- '<|im_end|>\\n' }}"""
chat_template = tokenizer.chat_template
chat_template = chat_template.replace(tokenizer_jinja_old, tokenizer_jinja_new)
tokenizer.chat_template = chat_template

"""Dataset Loading"""
train_set_name = "train"
val_set_name = "val"
test_set_name = "test"
input_name = "question"
label_name = "query"

features = Features({
    input_name: Value("string"),
    label_name: Value("string")
})

dataset = load_dataset("json", features=features, data_files={
    train_set_name: processed_train_file,
    val_set_name: processed_val_file
})

def train_and_eval_model(model, role_string, model_nickname, train_args=None, eval_args=None, lora_args=None):
    def format_chat(example):
        inputs = example[input_name]
        labels = example[label_name]

        batch_messages = []
        for input, label in zip(inputs, labels):
            if isinstance(input, list):
                input = str(input)
            if isinstance(label, list):
                label = str(label)
            
            batch_messages.append([
                {"role": "system", "content": role_string},
                {"role": "user", "content": input},
                {"role": "assistant", "content": label}
            ])

        return {"messages": batch_messages}
    
    train_dataset = dataset[train_set_name].map(format_chat, batched=True, remove_columns=dataset[train_set_name].column_names)
    val_dataset = dataset[val_set_name].map(format_chat, batched=True, remove_columns=dataset[val_set_name].column_names)
    
    if lora_args is not None:
        model = get_peft_model(model, lora_args)

    if train_args is not None:
        print(f"\nTraining {model_nickname}...")
        train_args.output_dir = f"./results-{model_nickname}"
        trainer = SFTTrainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.0
                )
            ]
        )
        model.train()
        # This is a built-in PEFT helper that ensures embedding layers 
        # properly retain gradients during gradient checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        trainer.train()

    if eval_args is not None:
        model.eval()
        eval_trainer = SFTTrainer(
            model=model,
            args=eval_args,
            train_dataset=train_dataset,  # This goes unused, but I have to pass a train_dataset to avoid a crash!
            eval_dataset=val_dataset,
            processing_class=tokenizer
        )
        print(f"\nEvaluating {model_nickname} on the validation set...")

        # Accuracy10
        print("Evaluating Accuracy...")
        with open(os.path.join(data_dir, data_file_name)) as f:
            valid_data = json.load(f)  # We need to know the ids for the validation data, so I need to load it from a separate file.
        accuracy = calculate_score_from_model(model, role_string, valid_data, tokenizer, reference_dir, reference_file_name, prediction_dir, f"{model_nickname}_{prediction_file_name}")
        print(accuracy)

        # Validation loss
        print("Evaluating Loss...")
        eval_results = eval_trainer.evaluate()
        print(eval_results)


"""Training and LoRA Arguments"""
# Shared arguments
fp16=True
max_length=2048
per_device_eval_batch_size=1
assistant_only_loss=True
report_to="none"
eval_save_strategy="epoch"
eval_save_steps=50

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = SFTConfig(
    num_train_epochs=6,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=per_device_eval_batch_size,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,

    eval_strategy=eval_save_strategy,
    eval_steps=eval_save_steps,

    save_strategy=eval_save_strategy,
    save_steps=eval_save_steps,

    learning_rate=1e-4,
    fp16=fp16,
    gradient_checkpointing=True,
    gradient_accumulation_steps=16,
    report_to=report_to,
    assistant_only_loss=assistant_only_loss,  # Argument for completion-only loss (so we only calculate loss on the sql query)
    max_length=max_length
)

evaluation_args = SFTConfig(
    per_device_eval_batch_size=per_device_eval_batch_size,
    report_to=report_to,
    fp16=fp16,
    assistant_only_loss=assistant_only_loss,
    max_length=max_length,
    remove_unused_columns=False
)

# Baseline 1
nickname = "Baseline-Model-1_No-Schema-No-Training"
train_and_eval_model(
    model=baseline_model,
    role_string=baseline_role_string,
    model_nickname=nickname,
    eval_args=evaluation_args
)

# Baseline 2
nickname = "Baseline-Model-2_Full-Schema-No-Training"
train_and_eval_model(
    model=baseline_model,
    role_string=schema_role_string,
    model_nickname=nickname,
    eval_args=evaluation_args
)
del baseline_model
gc.collect()
torch.cuda.empty_cache()

# Model 1
nickname = "Model-1_Training-Only-No-Schema"
model_1 = copy.deepcopy(model)
train_and_eval_model(
    model=model_1,
    role_string=baseline_role_string,
    model_nickname=nickname,
    train_args=training_args,
    lora_args=lora_config
)
del model_1
gc.collect()
torch.cuda.empty_cache()

# Model 2
nickname = "Model-2_Full-Schema-and-Training"
model_2 = model  # TODO: Deep copy if more models are added in the future
train_and_eval_model(
    model=model_2,
    role_string=schema_role_string,
    model_nickname=nickname,
    train_args=training_args,
    lora_args=lora_config
)
del model_2
gc.collect()
torch.cuda.empty_cache()