from graph_builder import build_mimiciv_knowledge_graph
from deterministic_knowledge_graph_retrieval import retrieve_schema_prompt
from rag_knowledge_graph_retrieval import get_retriever

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

get_retriever(schema_graph)

"""
question = "How is potassium chl 40 meq / 1000 ml d5ns delivered to the body?"
print(question)
context = retrieve_schema_prompt(question, schema_graph)
print(context)

question = "What are the standard methods used for ingesting 5% dextrose (excel bag)?"
print(question)
context = retrieve_schema_prompt(question, schema_graph)
print(context)
"""