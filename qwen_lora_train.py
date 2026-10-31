# import sys
# !{sys.executable} -m pip install "torch<2.5.0" "transformers>=4.42.0" huggingface_hub tiktoken einops transformers_stream_generator datasets accelerate peft ipywidgets trl

import os
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from datasets import load_dataset, Features, Value
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from inference_utils import calculate_score_from_model, get_role_string_with_knowledge_graph_context, TableSelectionFunction
from role_strings import ANSWERABLE_STRING, baseline_role_string, is_role_string_for_answerability_task, schema_role_string, answerability_role_string, pipeline_sql_role_string, UNANSWERABLE_STRING
from dataset_constants import input_name, label_name, processed_train_file, processed_val_file, data_dir, data_file_name, prediction_dir, prediction_file_name, reference_dir, reference_file_name
from graph_builder import build_mimiciv_knowledge_graph
from rag_knowledge_graph_retrieval import RetrievalSystem

"""Constants"""
TRAIN_SET_NAME = "train"
VAL_SET_NAME = "val"
TEST_SET_NAME = "test"

"""Model and Tokenizer Instantiation (and Role String + Jinja modifications)"""
model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# pad_token settings to prevent logs from getting flooded
tokenizer.pad_token = tokenizer.eos_token
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
features = Features({
    input_name: Value("string"),
    label_name: Value("string")
})

dataset = load_dataset("json", features=features, data_files={
    TRAIN_SET_NAME: processed_train_file,
    VAL_SET_NAME: processed_val_file
})

def format_chat_dataset(split, role_string, knowledge_graph_retrieval_system=None, relevancy_value=None, table_selection_function=None, exclude_unanswerable_questions=False):
    answerability_task = is_role_string_for_answerability_task(role_string)
    def format_chat(batch):
        inputs = batch[input_name]
        labels = batch[label_name]
    
        batch_messages = []
        for input, label in zip(inputs, labels):
            if isinstance(input, list):
                input = str(input)
            if isinstance(label, list):
                label = str(label)

            if (label == UNANSWERABLE_STRING and exclude_unanswerable_questions):
                continue
            elif (label != UNANSWERABLE_STRING and answerability_task):
                label = ANSWERABLE_STRING
    
            current_role_string = role_string
            if knowledge_graph_retrieval_system is not None:
                current_role_string = get_role_string_with_knowledge_graph_context(knowledge_graph_retrieval_system, role_string, input, relevancy_value, table_selection_function)
    
            batch_messages.append([
                {"role": "system", "content": current_role_string},
                {"role": "user", "content": input},
                {"role": "assistant", "content": label}
            ])
        return {"messages": batch_messages}
    return split.map(format_chat, batched=True, remove_columns=split.column_names)

def load_accuracy_valid_data():
    with open(os.path.join(data_dir, data_file_name)) as f:
        valid_data = json.load(f)  # We need to know the ids for the validation data, so I need to load it from a separate file.
    return valid_data

def train_and_eval_model(model, tokenizer, role_string, model_nickname, train_dataset=None, val_dataset=None, exclude_unanswerable_questions=False, knowledge_graph_retrieval_system=None, relevancy_value=None, table_selection_function=None, train_args=None, eval_args=None, lora_args=None):
    if train_dataset is None:
        train_dataset = format_chat_dataset(dataset[TRAIN_SET_NAME], role_string, knowledge_graph_retrieval_system, relevancy_value, table_selection_function, exclude_unanswerable_questions)
    if val_dataset is None:
        val_dataset = format_chat_dataset(dataset[VAL_SET_NAME], role_string, knowledge_graph_retrieval_system, relevancy_value, table_selection_function, exclude_unanswerable_questions)
    
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
        # This is a built-in PEFT helper that ensures embedding layers 
        # properly retain gradients during gradient checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        trainer.train()

        if trainer.state.best_model_checkpoint is not None:
            print(f"Best model checkpoint: {trainer.state.best_model_checkpoint}")
    else:
        print("Skipping Training. No train_args.")

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
        valid_data = load_accuracy_valid_data()
        accuracy = calculate_score_from_model([model], [role_string], valid_data, tokenizer, 
                                              reference_dir, reference_file_name, prediction_dir, f"{model_nickname}_{prediction_file_name}", 
                                              knowledge_graph_retrieval_system=knowledge_graph_retrieval_system, relevancy_value=relevancy_value, table_selection_function=table_selection_function)
        print("Accuracy Results:")
        print(accuracy)

        # Validation loss
        print("Evaluating Loss...")
        eval_results = eval_trainer.evaluate()
        print(eval_results)
    else:
        print("Skipping Eval. No eval_args.")
    return model

def create_knowledge_graph_retrieval_system(train_dataset=None):
    schema_graph = build_mimiciv_knowledge_graph()
    retrieval_system = RetrievalSystem(schema_graph)
    if train_dataset is not None:
        retrieval_system.train_embeddings(train_dataset, input_name, label_name)
    return retrieval_system

"""Get Model Nickname"""
parser = argparse.ArgumentParser()
parser.add_argument("--model_nickname", type=str, default="default_model")
args = parser.parse_args()
model_nickname = args.model_nickname

"""Training and LoRA Arguments"""
# Shared arguments
top_k_tables=5
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
    logging_dir=f"./logs/{model_nickname}",
    logging_steps=10,

    eval_strategy=eval_save_strategy,
    eval_steps=eval_save_steps,
    save_strategy=eval_save_strategy,
    save_steps=eval_save_steps,

    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,

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

def pipeline_train_and_eval(model, model_nickname, relevancy_value, table_selection_function):
    knowledge_graph_retrieval_system = create_knowledge_graph_retrieval_system(dataset[TRAIN_SET_NAME])
    first_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    second_model = model

    first_model = train_and_eval_model(
        model=first_model,
        tokenizer=tokenizer,
        role_string=answerability_role_string,
        model_nickname=model_nickname + "-FirstModel",
        exclude_unanswerable_questions=False,
        knowledge_graph_retrieval_system=knowledge_graph_retrieval_system,
        relevancy_value=relevancy_value,
        table_selection_function=table_selection_function,
        train_args=training_args,
        eval_args=None,
        lora_args=lora_config
    )
    second_model = train_and_eval_model(
        model=second_model,
        tokenizer=tokenizer,
        role_string=pipeline_sql_role_string,
        model_nickname=model_nickname + "-SecondModel",
        exclude_unanswerable_questions=True,
        knowledge_graph_retrieval_system=knowledge_graph_retrieval_system,
        relevancy_value=relevancy_value,
        table_selection_function=table_selection_function,
        train_args=training_args,
        eval_args=None,
        lora_args=lora_config
    )

    print("Evaluating Accuracy...")
    first_model.eval()
    second_model.eval()
    valid_data = load_accuracy_valid_data()
    accuracy = calculate_score_from_model([first_model, second_model], [answerability_role_string, pipeline_sql_role_string], valid_data, tokenizer, 
                                        reference_dir, reference_file_name, prediction_dir, f"{model_nickname}_{prediction_file_name}", 
                                        knowledge_graph_retrieval_system=knowledge_graph_retrieval_system, relevancy_value=relevancy_value, table_selection_function=table_selection_function)
    print("Accuracy Results:")
    print(accuracy)

# Change model to train depending on nickname
# First wave of tests: Do we need training, and how should we distribute schema information?
if model_nickname == "Baseline-Model-1_No-Schema-No-Training":
    train_and_eval_model(
        model=model,
        tokenizer=tokenizer,
        role_string=baseline_role_string,
        model_nickname=model_nickname,
        eval_args=evaluation_args
    )
elif model_nickname == "Baseline-Model-2_Full-Schema-No-Training":
    train_and_eval_model(
        model=model,
        tokenizer=tokenizer,
        role_string=schema_role_string,
        model_nickname=model_nickname,
        eval_args=evaluation_args
    )
elif model_nickname == "Baseline-Model-3_Untrained-Graph-Retriever-No-Model-Training":
    knowledge_graph_retrieval_system = create_knowledge_graph_retrieval_system()
    train_and_eval_model(
        model=model,
        tokenizer=tokenizer,
        role_string=baseline_role_string,
        model_nickname=model_nickname,
        knowledge_graph_retrieval_system=knowledge_graph_retrieval_system,
        relevancy_value=top_k_tables,
        table_selection_function=TableSelectionFunction.retrieve_top_k_tables,
        eval_args=evaluation_args
    )
elif model_nickname == "Baseline-Model-4_Trained-Graph-Retriever-No-Model-Training":
    knowledge_graph_retrieval_system = create_knowledge_graph_retrieval_system(dataset[TRAIN_SET_NAME])
    train_and_eval_model(
        model=model,
        tokenizer=tokenizer,
        role_string=baseline_role_string,
        model_nickname=model_nickname,
        knowledge_graph_retrieval_system=knowledge_graph_retrieval_system,
        relevancy_value=top_k_tables,
        table_selection_function=TableSelectionFunction.retrieve_top_k_tables,
        eval_args=evaluation_args
    )
elif model_nickname == "Model-1_Training-Only-No-Schema":
    train_and_eval_model(
        model=model,
        tokenizer=tokenizer,
        role_string=baseline_role_string,
        model_nickname=model_nickname,
        train_args=training_args,
        eval_args=evaluation_args,
        lora_args=lora_config
    )
elif model_nickname == "Model-2_Full-Schema-and-Training":
    train_and_eval_model(
        model=model,
        tokenizer=tokenizer,
        role_string=schema_role_string,
        model_nickname=model_nickname,
        train_args=training_args,
        eval_args=evaluation_args,
        lora_args=lora_config
    )
elif model_nickname == "Model-3_Untrained-Graph-Retriever-and-Model-Training":
    knowledge_graph_retrieval_system = create_knowledge_graph_retrieval_system()
    train_and_eval_model(
        model=model,
        tokenizer=tokenizer,
        role_string=baseline_role_string,
        model_nickname=model_nickname,
        knowledge_graph_retrieval_system=knowledge_graph_retrieval_system,
        relevancy_value=top_k_tables,
        table_selection_function=TableSelectionFunction.retrieve_top_k_tables,
        train_args=training_args,
        eval_args=evaluation_args,
        lora_args=lora_config
    )
elif model_nickname == "Model-4_Trained-Graph-Retriever-and-Model-Training":
    knowledge_graph_retrieval_system = create_knowledge_graph_retrieval_system(dataset[TRAIN_SET_NAME])
    train_and_eval_model(
        model=model,
        tokenizer=tokenizer,
        role_string=baseline_role_string,
        model_nickname=model_nickname,
        knowledge_graph_retrieval_system=knowledge_graph_retrieval_system,
        relevancy_value=top_k_tables,
        table_selection_function=TableSelectionFunction.retrieve_top_k_tables,
        train_args=training_args,
        eval_args=evaluation_args,
        lora_args=lora_config
    )
# Second wave of tests: Is it a good idea to use a score to decide how many tables to provide for a query? If so, what's a good score?
# Also, how does the model perform with a pipeline compared to having one model handle everything?
elif model_nickname == "Pipeline-Low_Relevancy":
    #pipeline_train_and_eval(model, "0.5_relevancy_threshold", 0.5, TableSelectionFunction.retrieve_relevant_tables)
    pipeline_train_and_eval(model, "3_tables", 3, TableSelectionFunction.retrieve_top_k_tables)
elif model_nickname == "Pipeline-Medium_Relevancy":
    #pipeline_train_and_eval(model, "0.6_relevancy_threshold", 0.6, TableSelectionFunction.retrieve_relevant_tables)
    pipeline_train_and_eval(model, "4_tables", 4, TableSelectionFunction.retrieve_top_k_tables)
elif model_nickname == "Pipeline-High_Relevancy":
    #pipeline_train_and_eval(model, "0.7_relevancy_threshold", 0.7, TableSelectionFunction.retrieve_relevant_tables)
    pipeline_train_and_eval(model, "5_tables", 5, TableSelectionFunction.retrieve_top_k_tables)
elif model_nickname == "Pipeline-Very-High_Relevancy":
    #pipeline_train_and_eval(model, "0.8_relevancy_threshold", 0.8, TableSelectionFunction.retrieve_relevant_tables)
    pipeline_train_and_eval(model, "6_tables", 6, TableSelectionFunction.retrieve_top_k_tables)
elif model_nickname == "Pipeline-Low_Tables":
    pipeline_train_and_eval(model, "3_tables", 3, TableSelectionFunction.retrieve_top_k_tables)
elif model_nickname == "Pipeline-Medium_Tables":
    pipeline_train_and_eval(model, "4_tables", 4, TableSelectionFunction.retrieve_top_k_tables)
elif model_nickname == "Pipeline-High_Tables":
    pipeline_train_and_eval(model, "5_tables", 5, TableSelectionFunction.retrieve_top_k_tables)
elif model_nickname == "Pipeline-Very-High_Tables":
    pipeline_train_and_eval(model, "6_tables", 6, TableSelectionFunction.retrieve_top_k_tables)
else:
    print("Model nickname unknown...")
print("Ending program...")