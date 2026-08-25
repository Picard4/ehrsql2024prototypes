from datasets import load_dataset, Features, Value

from graph_builder import build_mimiciv_knowledge_graph
from deterministic_knowledge_graph_retrieval import retrieve_schema_prompt
from rag_knowledge_graph_retrieval import RetrievalSystem
from dataset_constants import input_name, label_name, processed_train_file

schema_graph = build_mimiciv_knowledge_graph()

def print_graph(schema_graph):
    print("=== TABLES ===")
    for table, data in schema_graph.nodes(data=True):
        print(f"\n{table}")
        for column, details in data.get("columns", {}).items():
            print(f"  {column}: {details}")

    print("\n=== RELATIONSHIPS ===")
    for source, target, data in schema_graph.edges(data=True):
        print(
            f"{source}.{data.get('source_column')} "
            f"--[{data.get('relationship')}]--> "
            f"{target}.{data.get('target_column')}"
        )

retrieval_system = RetrievalSystem(schema_graph)

questions = [
    "How is potassium chl 40 meq / 1000 ml d5ns delivered to the body?",  # Train
    "What are the standard methods used for ingesting 5% dextrose (excel bag)?",  # Train
    "What does it cost to have catheter based invasive electrophysiologic testing?",  # Train
    "What is the price of diagnosis of dependence on respirator [ventilator] status?",  # Valid
    "Do you know patient 10005866's gender?",  # Valid
    "How much do patient 10018081's weight shifts measured at 2100-12-31 00:00:00 compared to the value measured at 2100-12-30 00:00:00?"  # Valid
]

print("Untrained Retrieval")
for question in questions:
    print(question)
    top_k_tables = retrieval_system.retrieve_top_k_tables(question)
    print(top_k_tables)
    print("\n")

features = Features({
    input_name: Value("string"),
    label_name: Value("string")
})

dataset = load_dataset("json", features=features, data_files={
    "train": processed_train_file
})
retrieval_system.train_embeddings(dataset["train"], input_name, label_name)

print("Trained Retrieval")
for question in questions:
    print(question)
    top_k_tables = retrieval_system.retrieve_top_k_tables(question)
    print(top_k_tables)
    print("\n")