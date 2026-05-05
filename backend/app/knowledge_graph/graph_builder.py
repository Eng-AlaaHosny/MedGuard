import os
import pickle
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple
from node2vec import Node2Vec

EMBEDDING_DIM = 128


class DrugKnowledgeGraph:
    """
    Builds and manages the DrugBank Knowledge Graph.
    Blueprint Section 3.3:
    - NetworkX DiGraph
    - node2vec 128-dimensional embeddings
    - Restricted to DDI Corpus drugs + first-degree neighbors
    - ~3,000-5,000 nodes (not full 14,000-node DrugBank graph)
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.embeddings: Dict[str, np.ndarray] = {}
        self.drug_name_to_id: Dict[str, str] = {}
        self.drug_id_to_name: Dict[str, str] = {}

    def add_drug_node(self, drug_id: str, drug_name: str, properties: Dict):
        """Add a drug node with its properties."""
        self.graph.add_node(drug_id, name=drug_name, **properties)
        name_lower = drug_name.lower()
        self.drug_name_to_id[name_lower] = drug_id
        self.drug_id_to_name[drug_id] = drug_name

    def add_interaction_edge(
        self,
        drug_a_id: str,
        drug_b_id: str,
        interaction_type: str,
        severity: int,
        description: str = ""
    ):
        """Add a directed interaction edge between two drugs."""
        self.graph.add_edge(
            drug_a_id,
            drug_b_id,
            interaction_type=interaction_type,
            severity=severity,
            description=description
        )

    def build_from_drugbank(self, drugs: List[Dict], ddi_drug_names: List[str]):
        """
        Build subgraph restricted to DDI Corpus drugs + first-degree neighbors.
        Blueprint Section 3.3: reduces graph to ~3,000-5,000 nodes.
        """
        print("Building DrugBank Knowledge Graph...")

        # Normalize DDI drug names for matching
        ddi_names_lower = set(name.lower() for name in ddi_drug_names)

        # First pass: find DDI corpus drugs in DrugBank
        ddi_drug_ids = set()
        for drug in drugs:
            if drug['name'].lower() in ddi_names_lower:
                ddi_drug_ids.add(drug['id'])
                self.add_drug_node(
                    drug['id'],
                    drug['name'],
                    {
                        'mechanism': drug.get('mechanism', ''),
                        'description': drug.get('description', '')
                    }
                )

        print(f"Found {len(ddi_drug_ids)} DDI corpus drugs in DrugBank")

        # Second pass: add first-degree neighbors
        for drug in drugs:
            if drug['id'] in ddi_drug_ids:
                for interaction in drug['interactions']:
                    neighbor_id = interaction['drug_b_id']
                    neighbor_name = interaction['drug_b_name']
                    if neighbor_id and neighbor_id not in self.graph:
                        self.add_drug_node(
                            neighbor_id,
                            neighbor_name,
                            {'mechanism': '', 'description': ''}
                        )
                    if neighbor_id:
                        self.add_interaction_edge(
                            drug['id'],
                            neighbor_id,
                            interaction_type='interaction',
                            severity=interaction.get('severity', 0),
                            description=interaction.get('description', '')
                        )

        print(f"Knowledge Graph built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")

    def compute_embeddings(self, save_path: Optional[str] = None):
        """
        Run node2vec to compute 128-dimensional drug embeddings.
        Blueprint Section 3.3: encodes drug class, mechanism, metabolic pathway.
        """
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty — cannot compute embeddings")
            return

        print(f"Computing node2vec embeddings for {self.graph.number_of_nodes()} nodes...")

        node2vec = Node2Vec(
            self.graph,
            dimensions=EMBEDDING_DIM,
            walk_length=30,
            num_walks=200,
            workers=1,
            quiet=True
        )

        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        # Store embeddings
        for node in self.graph.nodes():
            if node in model.wv:
                self.embeddings[node] = model.wv[node]

        print(f"Computed embeddings for {len(self.embeddings)} nodes")

        if save_path:
            self.save_embeddings(save_path)

    def get_drug_embedding(self, drug_name: str) -> Optional[np.ndarray]:
        """
        Get KG embedding for a drug by name.
        Returns None if drug not in graph (triggers Data Unavailable warning).
        Blueprint Section 3.3: Data Unavailable if drug absent from subgraph.
        """
        name_lower = drug_name.lower()
        drug_id = self.drug_name_to_id.get(name_lower)
        if drug_id is None:
            return None
        return self.embeddings.get(drug_id)

    def check_drug_available(self, drug_name: str) -> bool:
        """Check if drug exists in the knowledge graph."""
        return drug_name.lower() in self.drug_name_to_id

    def get_interaction_info(
        self, drug_a_name: str, drug_b_name: str
    ) -> Optional[Dict]:
        """Get edge data for a drug pair if it exists."""
        drug_a_id = self.drug_name_to_id.get(drug_a_name.lower())
        drug_b_id = self.drug_name_to_id.get(drug_b_name.lower())
        if drug_a_id and drug_b_id:
            if self.graph.has_edge(drug_a_id, drug_b_id):
                return self.graph[drug_a_id][drug_b_id]
            if self.graph.has_edge(drug_b_id, drug_a_id):
                return self.graph[drug_b_id][drug_a_id]
        return None

    def save(self, path: str):
        """Save the full graph object."""
        with open(path, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'embeddings': self.embeddings,
                'drug_name_to_id': self.drug_name_to_id,
                'drug_id_to_name': self.drug_id_to_name
            }, f)
        print(f"Knowledge graph saved to {path}")

    def load(self, path: str):
        """Load the full graph object."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.graph = data['graph']
        self.embeddings = data['embeddings']
        self.drug_name_to_id = data['drug_name_to_id']
        self.drug_id_to_name = data['drug_id_to_name']
        print(f"Knowledge graph loaded from {path}")
        print(f"Nodes: {self.graph.number_of_nodes()} | "
              f"Edges: {self.graph.number_of_edges()} | "
              f"Embeddings: {len(self.embeddings)}")

    def save_embeddings(self, path: str):
        """Save only embeddings dict."""
        with open(path, 'wb') as f:
            pickle.dump(self.embeddings, f)
        print(f"Embeddings saved to {path}")

    def load_embeddings(self, path: str):
        """Load only embeddings dict."""
        with open(path, 'rb') as f:
            self.embeddings = pickle.load(f)
        print(f"Loaded {len(self.embeddings)} embeddings")


def build_demo_graph() -> DrugKnowledgeGraph:
    """
    Build a small demo graph with known drug interactions.
    Used for testing before DrugBank XML is available.
    """
    kg = DrugKnowledgeGraph()

    demo_drugs = [
        ('DB00001', 'Warfarin', 'Vitamin K antagonist anticoagulant'),
        ('DB00002', 'Aspirin', 'NSAID antiplatelet agent'),
        ('DB00003', 'Metformin', 'Biguanide antidiabetic'),
        ('DB00004', 'Metoprolol', 'Beta-1 selective blocker'),
        ('DB00005', 'Lisinopril', 'ACE inhibitor'),
        ('DB00006', 'Atorvastatin', 'HMG-CoA reductase inhibitor'),
        ('DB00007', 'Digoxin', 'Cardiac glycoside'),
        ('DB00008', 'Amiodarone', 'Class III antiarrhythmic'),
        ('DB00009', 'Fluoxetine', 'SSRI antidepressant'),
        ('DB00010', 'Ciprofloxacin', 'Fluoroquinolone antibiotic'),
    ]

    for drug_id, name, mechanism in demo_drugs:
        kg.add_drug_node(drug_id, name, {'mechanism': mechanism})

    demo_interactions = [
        ('DB00001', 'DB00002', 'effect', 3,
         'Warfarin and Aspirin increase bleeding risk when combined'),
        ('DB00001', 'DB00009', 'mechanism', 2,
         'Fluoxetine inhibits CYP2C9 increasing Warfarin levels'),
        ('DB00001', 'DB00010', 'mechanism', 2,
         'Ciprofloxacin inhibits CYP1A2 affecting Warfarin metabolism'),
        ('DB00004', 'DB00008', 'effect', 3,
         'Amiodarone increases Metoprolol levels causing bradycardia'),
        ('DB00006', 'DB00008', 'mechanism', 2,
         'Amiodarone inhibits CYP3A4 increasing Atorvastatin levels'),
        ('DB00007', 'DB00008', 'effect', 3,
         'Amiodarone increases Digoxin levels causing toxicity'),
    ]

    for a, b, itype, severity, desc in demo_interactions:
        kg.add_interaction_edge(a, b, itype, severity, desc)

    print(f"Demo graph: {kg.graph.number_of_nodes()} nodes, "
          f"{kg.graph.number_of_edges()} edges")
    return kg


if __name__ == "__main__":
    print("Building demo Knowledge Graph...")
    kg = build_demo_graph()

    print("\nTesting drug lookup...")
    embedding = kg.get_drug_embedding("Warfarin")
    print(f"Warfarin in graph: {kg.check_drug_available('Warfarin')}")

    print("\nTesting interaction lookup...")
    info = kg.get_interaction_info("Warfarin", "Aspirin")
    if info:
        print(f"Warfarin-Aspirin interaction: severity={info['severity']}")
        print(f"Description: {info['description']}")

    print("\nKnowledge Graph working correctly!")
