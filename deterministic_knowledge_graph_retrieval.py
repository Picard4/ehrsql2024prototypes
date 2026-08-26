# This code is NOT used. It was just a little experiment I tried before realizing that I need something better to retrieve from a knowledge graph.

TABLE_KEYWORDS = {
    "patients": [
        "patient", "patients", "gender", "date of birth", "birth",
        "death", "deceased"
    ],
    "admissions": [
        "admission", "admissions", "hospitalization", "hospital",
        "discharge", "insurance", "marital", "language"
    ],
    "d_icd_diagnoses": [
        "diagnosis", "diagnoses", "icd"
    ],
    "d_icd_procedures": [
        "procedure", "procedures", "icd"
    ],
    "d_labitems": [
        "label", "item", "lab", "laboratory", "test"
    ],
    "d_items": [
        "label", "item", "measurement", "chart item"
    ],
    "diagnoses_icd": [
        "diagnosis", "diagnoses", "disease", "condition", "icd"
    ],
    "procedures_icd": [
        "procedure", "procedures", "surgery", "icd"
    ],
    "labevents": [
        "lab", "laboratory", "test", "blood", "measurement"
    ],
    "prescriptions": [
        "prescription", "prescriptions", "drug", "medication",
        "medicine"
    ],
    "cost": [
        "cost", "costs", "charge", "charges", "expense"
    ],
    "icustays": [
        "icu", "intensive care"
    ],
    "chartevents": [
        "chart", "vital", "vitals", "measurement"
    ],
    "inputevents": [
        "input", "fluid", "infusion", "administered"
    ],
    "outputevents": [
        "output", "urine", "urinary"
    ],
    "microbiologyevents": [
        "microbiology", "organism", "culture"
    ],
    "transfers": [
        "transfer", "transfers", "care unit"
    ]
}

def schema_to_prompt(schema_graph, table_names):
    sections = []

    for table_name in table_names:
        node = schema_graph.nodes[table_name]
        columns = node["columns"]

        column_lines = []

        for column_name, column_info in columns.items():
            column_lines.append(
                f"    {column_name} {column_info['type']}"
            )

        sections.append(
            f"TABLE {table_name}\n" +
            "\n".join(column_lines)
        )

    return "\n\n".join(sections)

def get_relevant_tables(question):
    question_lower = question.lower()

    relevant_tables = set()

    for table_name, keywords in TABLE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in question_lower:
                relevant_tables.add(table_name)
                break

    return relevant_tables

def expand_relevant_tables(schema_graph, seed_tables, hops=1):
    relevant_tables = set(seed_tables)

    frontier = set(seed_tables)

    for _ in range(hops):
        next_frontier = set()

        for table in frontier:
            for neighbor in schema_graph.neighbors(table):
                if neighbor not in relevant_tables:
                    relevant_tables.add(neighbor)
                    next_frontier.add(neighbor)

        frontier = next_frontier

    return relevant_tables

def get_relevant_tables(question, schema_graph):
    seed_tables = set()

    question_lower = question.lower()

    for table_name, keywords in TABLE_KEYWORDS.items():
        if any(keyword in question_lower for keyword in keywords):
            seed_tables.add(table_name)

    # If nothing was detected, fall back to the whole schema
    if not seed_tables:
        return set(schema_graph.nodes)

    # Expand one hop through the knowledge graph
    return expand_relevant_tables(schema_graph, seed_tables, hops=1)

def relationships_to_prompt(schema_graph, table_names):
    lines = []

    for source in table_names:
        for target in schema_graph.neighbors(source):

            if target not in table_names:
                continue

            edge_data = schema_graph.get_edge_data(source, target)

            relationship = edge_data["relationship"]
            source_column = edge_data["source_column"]
            target_column = edge_data["target_column"]

            line = (
                f"{source}.{source_column} "
                f"-> {target}.{target_column} "
                f"[{relationship}]"
            )

            if relationship == "POLYMORPHIC_REFERENCE":
                line += (
                    f" WHEN {source}.{edge_data['discriminator_column']} "
                    f"= '{edge_data['discriminator_value']}'"
                )

            lines.append(line)

    return "\n".join(lines)

def retrieve_schema_prompt(question, schema_graph):
    relevant_tables = get_relevant_tables(question, schema_graph)

    schema_text = schema_to_prompt(
        schema_graph,
        relevant_tables
    )

    relationship_text = relationships_to_prompt(
        schema_graph,
        relevant_tables
    )

    return f"""Relevant database schema:

    {schema_text}

    Relationships:
    {relationship_text}
    """