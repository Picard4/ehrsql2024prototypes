import networkx as nx

def build_mimiciv_knowledge_graph():

    schema_graph = nx.DiGraph()

    # Add tables as nodes
    schema_graph.add_node(
        "patients",
        columns={
            "row_id": {
                "type": "INT",
                "primary_key": True,
                "nullable": False,
                "unique": True
            },
            "subject_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": True
            },
            "gender": {
                "type": "VARCHAR(5)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "dob": {
                "type": "TIMESTAMP(0)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "dod": {
                "type": "TIMESTAMP(0)",
                "primary_key": False,
                "nullable": True,
                "unique": False
            }
        }
    )

    schema_graph.add_node(
        "admissions",
        columns={
            "row_id": {
                "type": "INT",
                "primary_key": True,
                "nullable": False,
                "unique": True
            },
            "subject_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "hadm_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": True
            },
            "admittime": {
                "type": "TIMESTAMP(0)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "dischtime": {
                "type": "TIMESTAMP(0)",
                "primary_key": False,
                "nullable": True,
                "unique": False
            },
            "admission_type": {
                "type": "VARCHAR(50)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "admission_location": {
                "type": "VARCHAR(50)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "discharge_location": {
                "type": "VARCHAR(50)",
                "primary_key": False,
                "nullable": True,
                "unique": False
            },
            "insurance": {
                "type": "VARCHAR(255)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "language": {
                "type": "VARCHAR(10)",
                "primary_key": False,
                "nullable": True,
                "unique": False
            },
            "marital_status": {
                "type": "VARCHAR(50)",
                "primary_key": False,
                "nullable": True,
                "unique": False
            },
            "age": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            }
        }
    )

    schema_graph.add_node(
        "d_icd_diagnoses",
        columns={
            "row_id": {
                "type": "INT",
                "primary_key": True,
                "nullable": False,
                "unique": True
            },
            "icd_code": {
                "type": "VARCHAR(10)",
                "primary_key": False,
                "nullable": False,
                "unique": True
            },
            "long_title": {
                "type": "VARCHAR(255)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            }
        }
    )

    schema_graph.add_node(
        "d_icd_procedures",
        columns={
            "row_id": {
                "type": "INT",
                "primary_key": True,
                "nullable": False,
                "unique": True
            },
            "icd_code": {
                "type": "VARCHAR(10)",
                "primary_key": False,
                "nullable": False,
                "unique": True
            },
            "long_title": {
                "type": "VARCHAR(255)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            }
        }
    )

    schema_graph.add_node(
        "d_labitems",
        columns={
            "row_id": {
                "type": "INT",
                "primary_key": True,
                "nullable": False,
                "unique": True
            },
            "itemid": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": True
            },
            "label": {
                "type": "VARCHAR(200)",
                "primary_key": False,
                "nullable": True,
                "unique": False
            }
        }
    )

    schema_graph.add_node(
        "d_items",
        columns={
            "row_id": {
                "type": "INT",
                "primary_key": True,
                "nullable": False,
                "unique": True
            },
            "itemid": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": True
            },
            "label": {
                "type": "VARCHAR(200)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "abbreviation": {
                "type": "VARCHAR(200)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "linksto": {
                "type": "VARCHAR(50)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            }
        }
    )

    schema_graph.add_node(
        "diagnoses_icd",
        columns={
            "row_id": {
                "type": "INT",
                "primary_key": True,
                "nullable": False,
                "unique": True
            },
            "subject_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "hadm_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "icd_code": {
                "type": "VARCHAR(10)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "charttime": {
                "type": "TIMESTAMP(0)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            }
        }
    )

    schema_graph.add_node(
        "procedures_icd",
        columns={
            "row_id": {
                "type": "INT",
                "primary_key": True,
                "nullable": False,
                "unique": True
            },
            "subject_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "hadm_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "icd_code": {
                "type": "VARCHAR(10)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "charttime": {
                "type": "TIMESTAMP(0)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            }
        }
    )

    schema_graph.add_node(
        "labevents",
        columns={
            "row_id": {
                "type": "INT",
                "primary_key": True,
                "nullable": False,
                "unique": True
            },
            "subject_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "hadm_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "itemid": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "charttime": {
                "type": "TIMESTAMP(0)",
                "primary_key": False,
                "nullable": True,
                "unique": False
            },
            "valuenum": {
                "type": "DOUBLE PRECISION",
                "primary_key": False,
                "nullable": True,
                "unique": False
            },
            "valueuom": {
                "type": "VARCHAR(20)",
                "primary_key": False,
                "nullable": True,
                "unique": False
            }
        }
    )

    schema_graph.add_node(
        "prescriptions",
        columns={
            "row_id": {
                "type": "INT",
                "primary_key": True,
                "nullable": False,
                "unique": True
            },
            "subject_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "hadm_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "starttime": {
                "type": "TIMESTAMP(0)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "stoptime": {
                "type": "TIMESTAMP(0)",
                "primary_key": False,
                "nullable": True,
                "unique": False
            },
            "drug": {
                "type": "VARCHAR(255)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "dose_val_rx": {
                "type": "VARCHAR(100)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "dose_unit_rx": {
                "type": "VARCHAR(50)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "route": {
                "type": "VARCHAR(50)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            }
        }
    )

    schema_graph.add_node(
        "cost",
        columns={
            "row_id": {
                "type": "INT",
                "primary_key": True,
                "nullable": False,
                "unique": True
            },
            "subject_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "hadm_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "event_type": {
                "type": "VARCHAR(20)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "event_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "chargetime": {
                "type": "TIMESTAMP(0)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "cost": {
                "type": "DOUBLE PRECISION",
                "primary_key": False,
                "nullable": False,
                "unique": False
            }
        }
    )

    schema_graph.add_node(
        "icustays",
        columns={
            "row_id": {
                "type": "INT",
                "primary_key": True,
                "nullable": False,
                "unique": True
            },
            "subject_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "hadm_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "stay_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": True
            },
            "first_careunit": {
                "type": "VARCHAR(20)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "last_careunit": {
                "type": "VARCHAR(20)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "intime": {
                "type": "TIMESTAMP(0)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "outtime": {
                "type": "TIMESTAMP(0)",
                "primary_key": False,
                "nullable": True,
                "unique": False
            }
        }
    )

    schema_graph.add_node(
        "chartevents",
        columns={
            "row_id": {
                "type": "INT",
                "primary_key": True,
                "nullable": False,
                "unique": True
            },
            "subject_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "hadm_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "stay_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "itemid": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "charttime": {
                "type": "TIMESTAMP(0)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "valuenum": {
                "type": "DOUBLE PRECISION",
                "primary_key": False,
                "nullable": True,
                "unique": False
            },
            "valueuom": {
                "type": "VARCHAR(50)",
                "primary_key": False,
                "nullable": True,
                "unique": False
            }
        }
    )

    schema_graph.add_node(
        "inputevents",
        columns={
            "row_id": {
                "type": "INT",
                "primary_key": True,
                "nullable": False,
                "unique": True
            },
            "subject_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "hadm_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "stay_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "starttime": {
                "type": "TIMESTAMP(0)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "itemid": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "totalamount": {
                "type": "DOUBLE PRECISION",
                "primary_key": False,
                "nullable": True,
                "unique": False
            },
            "totalamountuom": {
                "type": "VARCHAR(50)",
                "primary_key": False,
                "nullable": True,
                "unique": False
            }
        }
    )

    schema_graph.add_node(
        "outputevents",
        columns={
            "row_id": {
                "type": "INT",
                "primary_key": True,
                "nullable": False,
                "unique": True
            },
            "subject_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "hadm_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "stay_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "charttime": {
                "type": "TIMESTAMP(0)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "itemid": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "value": {
                "type": "DOUBLE PRECISION",
                "primary_key": False,
                "nullable": True,
                "unique": False
            },
            "valueuom": {
                "type": "VARCHAR(50)",
                "primary_key": False,
                "nullable": True,
                "unique": False
            }
        }
    )

    schema_graph.add_node(
        "microbiologyevents",
        columns={
            "row_id": {
                "type": "INT",
                "primary_key": True,
                "nullable": False,
                "unique": True
            },
            "subject_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "hadm_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "charttime": {
                "type": "TIMESTAMP(0)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "spec_type_desc": {
                "type": "VARCHAR(100)",
                "primary_key": False,
                "nullable": True,
                "unique": False
            },
            "test_name": {
                "type": "VARCHAR(100)",
                "primary_key": False,
                "nullable": True,
                "unique": False
            },
            "org_name": {
                "type": "VARCHAR(100)",
                "primary_key": False,
                "nullable": True,
                "unique": False
            }
        }
    )

    schema_graph.add_node(
        "transfers",
        columns={
            "row_id": {
                "type": "INT",
                "primary_key": True,
                "nullable": False,
                "unique": True
            },
            "subject_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "hadm_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "transfer_id": {
                "type": "INT",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "eventtype": {
                "type": "VARCHAR(20)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "careunit": {
                "type": "VARCHAR(20)",
                "primary_key": False,
                "nullable": True,
                "unique": False
            },
            "intime": {
                "type": "TIMESTAMP(0)",
                "primary_key": False,
                "nullable": False,
                "unique": False
            },
            "outtime": {
                "type": "TIMESTAMP(0)",
                "primary_key": False,
                "nullable": True,
                "unique": False
            }
        }
    )

    # Add foreign-key relationships as edges

    # Admissions
    schema_graph.add_edge(
        "admissions", 
        "patients", 
        relationship="FOREIGN_KEY", 
        source_column="subject_id",
        target_column="subject_id"
    )

    # Procedures_ICD
    schema_graph.add_edge(
        "procedures_icd", 
        "d_icd_procedures", 
        relationship="FOREIGN_KEY", 
        source_column="icd_code",
        target_column="icd_code"
    )
    schema_graph.add_edge(
        "procedures_icd", 
        "admissions", 
        relationship="FOREIGN_KEY", 
        source_column="hadm_id",
        target_column="hadm_id"
    )
    schema_graph.add_edge(
        "procedures_icd", 
        "patients", 
        relationship="JOINABLE", 
        source_column="subject_id",
        target_column="subject_id"
    )

    # Diagnoses_ICD
    schema_graph.add_edge(
        "diagnoses_icd", 
        "d_icd_diagnoses", 
        relationship="FOREIGN_KEY", 
        source_column="icd_code",
        target_column="icd_code"
    )
    schema_graph.add_edge(
        "diagnoses_icd", 
        "admissions", 
        relationship="FOREIGN_KEY", 
        source_column="hadm_id",
        target_column="hadm_id"
    )
    schema_graph.add_edge(
        "diagnoses_icd", 
        "patients", 
        relationship="JOINABLE", 
        source_column="subject_id",
        target_column="subject_id"
    )

    # Prescriptions
    schema_graph.add_edge(
        "prescriptions", 
        "admissions", 
        relationship="FOREIGN_KEY", 
        source_column="hadm_id",
        target_column="hadm_id"
    )
    schema_graph.add_edge(
        "prescriptions", 
        "patients", 
        relationship="JOINABLE", 
        source_column="subject_id",
        target_column="subject_id"
    )

    # Cost
    schema_graph.add_edge(
        "cost", 
        "diagnoses_icd", 
        relationship="POLYMORPHIC_FOREIGN_KEY", 
        source_column="event_id",
        target_column="row_id",
        discriminator_column="event_type",
        discriminator_value="diagnoses_icd"
    )
    schema_graph.add_edge(
        "cost", 
        "procedures_icd", 
        relationship="POLYMORPHIC_FOREIGN_KEY", 
        source_column="event_id",
        target_column="row_id",
        discriminator_column="event_type",
        discriminator_value="procedures_icd"
    )
    schema_graph.add_edge(
        "cost", 
        "labevents", 
        relationship="POLYMORPHIC_FOREIGN_KEY", 
        source_column="event_id",
        target_column="row_id",
        discriminator_column="event_type",
        discriminator_value="labevents"
    )
    schema_graph.add_edge(
        "cost", 
        "prescriptions", 
        relationship="POLYMORPHIC_FOREIGN_KEY", 
        source_column="event_id",
        target_column="row_id",
        discriminator_column="event_type",
        discriminator_value="prescriptions"
    )
    schema_graph.add_edge(
        "cost", 
        "admissions", 
        relationship="FOREIGN_KEY", 
        source_column="hadm_id",
        target_column="hadm_id"
    )
    schema_graph.add_edge(
        "cost", 
        "patients", 
        relationship="JOINABLE", 
        source_column="subject_id",
        target_column="subject_id"
    )

    # Transfers
    schema_graph.add_edge(
        "transfers", 
        "admissions", 
        relationship="FOREIGN_KEY", 
        source_column="hadm_id",
        target_column="hadm_id"
    )
    schema_graph.add_edge(
        "transfers", 
        "patients", 
        relationship="JOINABLE", 
        source_column="subject_id",
        target_column="subject_id"
    )

    # MicrobiologyEvents
    schema_graph.add_edge(
        "microbiologyevents", 
        "admissions", 
        relationship="FOREIGN_KEY", 
        source_column="hadm_id",
        target_column="hadm_id"
    )
    schema_graph.add_edge(
        "microbiologyevents", 
        "patients", 
        relationship="JOINABLE", 
        source_column="subject_id",
        target_column="subject_id"
    )

    # LabEvents
    schema_graph.add_edge(
        "labevents", 
        "d_labitems", 
        relationship="FOREIGN_KEY", 
        source_column="itemid",
        target_column="itemid"
    )
    schema_graph.add_edge(
        "labevents", 
        "admissions", 
        relationship="FOREIGN_KEY", 
        source_column="hadm_id",
        target_column="hadm_id"
    )
    schema_graph.add_edge(
        "labevents", 
        "patients", 
        relationship="JOINABLE", 
        source_column="subject_id",
        target_column="subject_id"
    )

    # IcuStays
    schema_graph.add_edge(
        "icustays", 
        "admissions", 
        relationship="FOREIGN_KEY", 
        source_column="hadm_id",
        target_column="hadm_id"
    )
    schema_graph.add_edge(
        "icustays", 
        "patients", 
        relationship="JOINABLE", 
        source_column="subject_id",
        target_column="subject_id"
    )

    # ChartEvents
    schema_graph.add_edge(
        "chartevents", 
        "d_items", 
        relationship="FOREIGN_KEY", 
        source_column="itemid",
        target_column="itemid"
    )
    schema_graph.add_edge(
        "chartevents", 
        "icustays", 
        relationship="FOREIGN_KEY", 
        source_column="stay_id",
        target_column="stay_id"
    )
    schema_graph.add_edge(
        "chartevents", 
        "admissions", 
        relationship="FOREIGN_KEY", 
        source_column="hadm_id",
        target_column="hadm_id"
    )
    schema_graph.add_edge(
        "chartevents", 
        "patients", 
        relationship="JOINABLE", 
        source_column="subject_id",
        target_column="subject_id"
    )

    # InputEvents
    schema_graph.add_edge(
        "inputevents", 
        "d_items", 
        relationship="FOREIGN_KEY", 
        source_column="itemid",
        target_column="itemid"
    )
    schema_graph.add_edge(
        "inputevents", 
        "icustays", 
        relationship="FOREIGN_KEY", 
        source_column="stay_id",
        target_column="stay_id"
    )
    schema_graph.add_edge(
        "inputevents", 
        "admissions", 
        relationship="FOREIGN_KEY", 
        source_column="hadm_id",
        target_column="hadm_id"
    )
    schema_graph.add_edge(
        "inputevents", 
        "patients", 
        relationship="JOINABLE", 
        source_column="subject_id",
        target_column="subject_id"
    )

    # OutputEvents
    schema_graph.add_edge(
        "outputevents", 
        "d_items", 
        relationship="FOREIGN_KEY", 
        source_column="itemid",
        target_column="itemid"
    )
    schema_graph.add_edge(
        "outputevents", 
        "icustays", 
        relationship="FOREIGN_KEY", 
        source_column="stay_id",
        target_column="stay_id"
    )
    schema_graph.add_edge(
        "outputevents", 
        "admissions", 
        relationship="FOREIGN_KEY", 
        source_column="hadm_id",
        target_column="hadm_id"
    )
    schema_graph.add_edge(
        "outputevents", 
        "patients", 
        relationship="JOINABLE", 
        source_column="subject_id",
        target_column="subject_id"
    )

    return schema_graph