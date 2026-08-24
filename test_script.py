from graph_builder import build_mimiciv_knowledge_graph
from deterministic_knowledge_graph_retrieval import retrieve_schema_prompt
from rag_knowledge_graph_retrieval import RetrievalSystem

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

question = "How is potassium chl 40 meq / 1000 ml d5ns delivered to the body?"
print(question)
top_k_tables = retrieval_system.retrieve_top_k_tables(question, schema_graph)
print(top_k_tables)

question = "What are the standard methods used for ingesting 5% dextrose (excel bag)?"
print(question)
top_k_tables = retrieval_system.retrieve_top_k_tables(question, schema_graph)
print(top_k_tables)