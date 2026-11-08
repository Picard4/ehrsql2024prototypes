import numpy as np
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pickle
import os
import random

table_key = "table"
text_key = "text"
score_key = "score"

STATE_FILE_NAME = "retriever_state.pkl"
MODEL_FILE_NAME = "embedding_model"

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class RetrievalSystem:
    def __init__(self, schema_graph):
        self.schema_graph = schema_graph
        self.document_embeddings = None
        self.documents = []
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

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
                text_key: "\n".join(lines)
            })
    
        texts = [document[text_key] for document in self.documents]

        self.document_embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True
        )

    def train_embeddings(self, train_data, input_name, label_name, seed=42):
        # Ensure consistency by using a generator with a set seed.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)

        most_tables_in_a_query = 0  # Useful metric to know what top_k to use. Seems like 5 is a good number.
        table_documents = {
            document["table"]: document[text_key]
            for document in self.documents
        }

        table_names = list(self.schema_graph.nodes)
        train_examples = []

        # Determine and record which tables are and are not relevant to each query
        for example in train_data:
            relevant_tables = []
            irrelevant_tables = []

            for table in table_names:
                if table in example[label_name]:
                    relevant_tables.append(table)
                else:
                    irrelevant_tables.append(table)

            for relevant_table in relevant_tables:
                train_examples.append(
                    InputExample(
                        texts=[
                            example[input_name],
                            table_documents[relevant_table]
                        ],
                        label=1.0
                    )
                )

            most_tables_in_a_query = max(
                most_tables_in_a_query,
                len(relevant_tables)
            )

            for irrelevant_table in irrelevant_tables:
                train_examples.append(
                    InputExample(
                        texts=[
                            example[input_name],
                            table_documents[irrelevant_table]
                        ],
                        label=0.0
                    )
                )

        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=16,
            generator=generator,
            worker_init_fn=seed_worker
        )

        train_loss = losses.CosineSimilarityLoss(
            self.embedding_model
        )

        # Fit embedding model
        self.embedding_model.fit(
            train_objectives=[
                (train_dataloader, train_loss)
            ],
            epochs=1,
            warmup_steps=100,
            show_progress_bar=True
        )

        # Re-encode the document embeddings
        texts = [document[text_key] for document in self.documents]
        self.document_embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True
        )
        return most_tables_in_a_query

    def get_scores(self, question):
        question_embedding = self.embedding_model.encode(
            question,
            normalize_embeddings=True
        )
        return self.document_embeddings @ question_embedding

    def get_chosen_tables(self, scores, top_indices):
        return [
            {
                table_key: self.documents[i][table_key],
                text_key: self.documents[i][text_key],
                score_key: float(scores[i])
            }
            for i in top_indices
        ]

    def retrieve_top_k_tables(self, question, top_k=5):
        scores = self.get_scores(question)
        top_indices = scores.argsort()[::-1][:top_k]
        return self.get_chosen_tables(scores, top_indices)

    def retrieve_relevant_tables(self, question, relevancy_threshold=0.5):
        scores = self.get_scores(question)
        top_indices = np.flatnonzero(scores >= relevancy_threshold)
        if len(top_indices) == 0:
            # Get the highest-scoring table if no tables were eligible
            top_indices = np.array([scores.argmax()])
        else:
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return self.get_chosen_tables(scores, top_indices)

    def save(self, save_dir="./retriever_saved"):
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Save the underlying SentenceTransformer model natively
        self.embedding_model.save(os.path.join(save_dir, MODEL_FILE_NAME))
        
        # 2. Save documents, embeddings, and schema graph using pickle
        state = {
            "documents": self.documents,
            "document_embeddings": self.document_embeddings,
            "schema_graph": self.schema_graph
        }
        with open(os.path.join(save_dir, STATE_FILE_NAME), "wb") as f:
            pickle.dump(state, f)
        print(f"RetrievalSystem saved to {save_dir}")

    @classmethod
    def load(cls, save_dir="./retriever_saved"):
        # 1. Load state metadata
        with open(os.path.join(save_dir, STATE_FILE_NAME), "rb") as f:
            state = pickle.load(f)
            
        # 2. Instantiate object bypassing __init__ or re-initializing lightly
        instance = cls.__new__(cls)
        instance.schema_graph = state["schema_graph"]
        instance.documents = state["documents"]
        instance.document_embeddings = state["document_embeddings"]
        
        # 3. Load the SentenceTransformer model
        instance.embedding_model = SentenceTransformer(os.path.join(save_dir, MODEL_FILE_NAME))
        
        print(f"RetrievalSystem loaded from {save_dir}")
        return instance