import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.data.drugbank_processor import parse_drugbank_xml
from app.knowledge_graph.graph_builder import DrugKnowledgeGraph
from app.data.preprocessor import load_ddi_corpus


def build_full_kg(data_dir: str):
    """
    Build the full Knowledge Graph from DrugBank.
    Blueprint Section 3.3: restricted to DDI Corpus drugs + first-degree neighbors.
    """

    # Step 1: Load DDI Corpus to get drug names
    print("Step 1: Loading DDI Corpus drug names...")
    corpus_dir = os.path.join(data_dir, 'DDICorpus')
    train_sents, _ = load_ddi_corpus(corpus_dir)

    # Extract all drug names from DDI Corpus
    ddi_drug_names = set()
    for sent in train_sents:
        for entity in sent.entities:
            if entity.text:
                ddi_drug_names.add(entity.text.lower())

    print(f"Found {len(ddi_drug_names)} unique drug names in DDI Corpus")

    # Step 2: Parse DrugBank XML
    print("\nStep 2: Parsing DrugBank XML...")
    xml_path = os.path.join(data_dir, 'drugbank_full.xml', 'full database.xml')
    drugs = parse_drugbank_xml(xml_path)

    # Step 3: Build Knowledge Graph
    print("\nStep 3: Building Knowledge Graph...")
    kg = DrugKnowledgeGraph()
    kg.build_from_drugbank(drugs, list(ddi_drug_names))

    print(f"\nKnowledge Graph stats:")
    print(f"  Nodes: {kg.graph.number_of_nodes()}")
    print(f"  Edges: {kg.graph.number_of_edges()}")

    # Step 4: Compute node2vec embeddings
    print("\nStep 4: Computing node2vec embeddings...")
    embeddings_path = os.path.join(data_dir, 'kg_embeddings.pkl')
    kg.compute_embeddings(save_path=embeddings_path)

    # Step 5: Save full graph
    print("\nStep 5: Saving Knowledge Graph...")
    graph_path = os.path.join(data_dir, 'knowledge_graph.pkl')
    kg.save(graph_path)

    print("\n✅ Full Knowledge Graph built successfully!")
    print(f"Graph saved to: {graph_path}")
    print(f"Embeddings saved to: {embeddings_path}")

    return kg


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    build_full_kg(data_dir)
