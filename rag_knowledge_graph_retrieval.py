from sentence_transformers import SentenceTransformer

class RetrievalSystem:
    def __init__(self, schema_graph):
        self.schema_graph = schema_graph
        self.document_embeddings = None
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.documents = []

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

            self.documents.append({
                "table": table,
                "text": "\n".join(lines)
            })

        print(len(self.documents))
        print([doc["table"] for doc in self.documents])
        for doc in self.documents:
            print("=" * 50)
            print(doc["text"])
        texts = [document["text"] for document in self.documents]

        self.document_embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True
        )

    def retrieve_top_k_tables(self, question, top_k=5):
        question_embedding = self.embedding_model.encode(
            question,
            normalize_embeddings=True
        )

        scores = self.document_embeddings @ question_embedding

        top_indices = scores.argsort()[::-1][:top_k]

        return [
            {
                "table": self.documents[i]["table"],
                "text": self.documents[i]["text"],
                "score": float(scores[i])
            }
            for i in top_indices
        ]

    def train_embeddings(self, train_data, input_name, label_name):
        table_names = list(self.schema_graph.nodes)
        print(table_names)
        for example in train_data:
            relevant_tables = []
            irrelevant_tables = []
            for table in table_names:
                if table in example[label_name]:
                    relevant_tables.append(table)
                else:
                    irrelevant_tables.append(table)
