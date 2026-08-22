"""Load structured clinical mock records used to build the RDF graph and ground LLM synthesis."""
import json
import os


def load_mock_sources():
    """Read data/mock_sources.json from the project root (CSR, protocols, safety, scientific responses)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print('base dir ................',base_dir)
    file_path = os.path.join(base_dir, "data", "mock_sources.json")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Imported by ontology.py (graph build) and agent.py (synthesis context).
MOCK_SOURCES = load_mock_sources()