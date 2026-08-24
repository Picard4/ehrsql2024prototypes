from sentence_transformers import SentenceTransformer

def get_retriever(schema_graph):
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    documents = []

    for table, data in schema_graph.nodes(data=True):
        lines = [f"TABLE {table}"]

    lines.append("Columns:")
    for column, details in data.get("columns", {}).items():
        lines.append(
            f"  {column}: {details['type']}"
        )

    lines.append("Relationships:")
    for source, target, edge_data in schema_graph.edges(data=True):
        if source == table:
            if edge_data.get('relationship').lower() == "polymorphic_foreign_key":
                lines.append(
                    f"  {source}.{edge_data.get('source_column')} "
                    f"references "
                    f"{target}.{edge_data.get('target_column')} "
                    f"when {edge_data.get('discriminator_column')}='{edge_data.get('discriminator_value')}'"
                )
            else:    
                lines.append(
                    f"  {source}.{edge_data.get('source_column')} "
                    f"references "
                    f"{target}.{edge_data.get('target_column')}"
                )

    documents.append({
        "table": table,
        "text": "\n".join(lines)
    })

    print("ALL DOCUMENTS")
    print(documents)
    texts = [document["text"] for document in documents]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    )