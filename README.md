# MedReconGNN: Medication Reconciliation with GNN and Optimization

This repository provides a full pipeline for detecting discrepancies in patient medication lists and generating optimized reconciled drug recommendations using Graph Neural Networks (GNNs), knowledge graph querying, and mathematical optimization.

## 📂 Project Structure

- `medReconGNN.py`: GNN-based embedding generator using PyTorch Geometric on heterogeneous drug-disease graphs.
- `optimizer.py`: Gurobi-based multi-objective optimization to determine optimal and reconciled drug lists.
- `paramGenerator.py`: Interfaces with Neo4j and drives the full pipeline: querying, discrepancy detection, candidate generation, and optimization.
- `load_med.cypher`: Cypher script to load the medical knowledge graph into Neo4j.
- `data/`: Directory containing processed graph data, patient records, embeddings, and mapping files.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Neo4j (tested on 5.x)
- Gurobi (with valid license)
- Required Python packages:
    ```bash
    pip install torch torch-geometric pandas py2neo gurobipy matplotlib seaborn openpyxl
    ```

### 1. Load the Knowledge Graph

Start your Neo4j database and run:

```bash
cat load_med.cypher | cypher-shell -u neo4j -p <your-password>
```

### 2. Train GNN Embeddings
```bash
    python medReconGNN.py 
```
This will train a GAT-based model and save drug node embeddings.

### 3. Run the Full Pipeline
```bash
    python paramGenerator.py 
```

This script:

- Loads a labeled patient dataset.

- Queries the Neo4j graph for treatment/contraindication data.

- Uses Gurobi to solve for optimal and reconciled medication lists.

- Exports results to reconciled_list.xlsx and reconciled_list.pkl.

## 📌 Notes
- Drug and disease nodes are identified via DrugBank, ATC codes, UMLS CUIS, MONDO codes, and ICD9CM codes.

- Relationships: TREATS, HAS_CONTRAIND, INTERACTS, HAS_PARENTCODE, etc.

- Neo4j must be running locally with the database named combine2.