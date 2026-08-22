import os
import json
import gradio as gr
from typing import Dict, List, Any, TypedDict
from rdflib import Graph, Literal, RDF, Namespace, URIRef
from rdflib.namespace import XSD, RDFS

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

# -------------------------------------------------------------------
# 1. DEEP MULTI-LEVEL SOURCE DOCUMENTS (Expanded Safety & Sci-Res)
# -------------------------------------------------------------------
MOCK_SOURCES = {
    "Clinical Study Reports": [
        {
            "study_id": "NCT04561234",
            "title": "Phase 3 Study of Daratumumab in Relapsed Multiple Myeloma",
            "disease": {
                "term": "Multiple Myeloma",
                "therapeutic_area": "Hematological Malignancy",
                "snomed_code": "109989006"
            },
            "treatment": {
                "regimen": "Daratumumab 1800mg SubQ",
                "active_ingredient": "Daratumumab",
                "mechanism_of_action": "CD38-directed cytolytic antibody",
            }
        }
    ],
    "Protocols & Amendments": [
        {
            "study_id": "NCT04561234",
            "population": {
                "description": "Adults >= 18y",
                "biomarker": "CD38+ plasma cells >= 10%",
            }
        }
    ],
    "Safety / RIM Content": [
        {
            "study_id": "NCT04561234",
            "safety_report": {
                "report_id": "SR-9921-Q3",
                "adverse_event": "Severe Neutropenia",
                "meddra_pt": "Neutrophil count decreased",
                "meddra_soc": "Blood and lymphatic system disorders"
            }
        }
    ],
    "Scientific Responses": [
        {
            "study_id": "NCT04561234",
            "response_doc": {
                "doc_id": "SciRes-2026-0891",
                "inquiry_topic": "SubQ Administration in Renal Impairment",
                "medical_concept": "Pharmacokinetics (PK)"
            }
        }
    ]
}

# -------------------------------------------------------------------
# 2. ONTOLOGY BUILDER (Adding Deep Safety & Sci-Res Hierarchies)
# -------------------------------------------------------------------
CTI = Namespace("https://w3id.org/cti/ontology#")

def build_clinical_knowledge_graph() -> Graph:
    g = Graph()
    g.bind("cti", CTI)
    g.bind("rdfs", RDFS)

    # 1. Base Clinical Study + Disease + Treatment (from previous)
    for csr in MOCK_SOURCES["Clinical Study Reports"]:
        study_uri = URIRef(f"https://w3id.org/cti/study/{csr['study_id']}")
        g.add((study_uri, RDF.type, CTI.ClinicalStudy))
        g.add((study_uri, RDFS.label, Literal(csr["study_id"])))

        # Disease Hierarchy
        dis_uri = URIRef(f"https://w3id.org/cti/disease/{csr['disease']['snomed_code']}")
        ta_uri = URIRef(f"https://w3id.org/cti/ta/{csr['disease']['therapeutic_area'].replace(' ', '')}")
        g.add((dis_uri, RDF.type, CTI.DiseaseCondition))
        g.add((dis_uri, RDFS.label, Literal(csr["disease"]["term"])))
        g.add((ta_uri, RDF.type, CTI.TherapeuticArea))
        g.add((ta_uri, RDFS.label, Literal(csr["disease"]["therapeutic_area"])))
        g.add((study_uri, CTI.targetsDisease, dis_uri))
        g.add((dis_uri, CTI.belongsToTherapeuticArea, ta_uri))

        # Treatment Hierarchy
        tx_uri = URIRef(f"https://w3id.org/cti/treatment/{csr['study_id']}_tx")
        ai_uri = URIRef(f"https://w3id.org/cti/ingredient/{csr['treatment']['active_ingredient']}")
        moa_uri = URIRef(f"https://w3id.org/cti/moa/{csr['treatment']['mechanism_of_action'].replace(' ', '')}")
        g.add((tx_uri, RDF.type, CTI.Treatment))
        g.add((tx_uri, RDFS.label, Literal(csr["treatment"]["regimen"])))
        g.add((ai_uri, RDF.type, CTI.ActiveIngredient))
        g.add((ai_uri, RDFS.label, Literal(csr["treatment"]["active_ingredient"])))
        g.add((moa_uri, RDF.type, CTI.MechanismOfAction))
        g.add((moa_uri, RDFS.label, Literal(csr["treatment"]["mechanism_of_action"])))
        g.add((study_uri, CTI.usesTreatment, tx_uri))
        g.add((tx_uri, CTI.hasActiveIngredient, ai_uri))
        g.add((ai_uri, CTI.hasMechanismOfAction, moa_uri))

    # 2. Population Hierarchy
    for proto in MOCK_SOURCES["Protocols & Amendments"]:
        study_uri = URIRef(f"https://w3id.org/cti/study/{proto['study_id']}")
        pop_uri = URIRef(f"https://w3id.org/cti/population/{proto['study_id']}_pop")
        bio_uri = URIRef(f"https://w3id.org/cti/biomarker/CD38")
        g.add((pop_uri, RDF.type, CTI.Population))
        g.add((pop_uri, RDFS.label, Literal(proto["population"]["description"])))
        g.add((bio_uri, RDF.type, CTI.Biomarker))
        g.add((bio_uri, RDFS.label, Literal(proto["population"]["biomarker"])))
        g.add((study_uri, CTI.hasPopulation, pop_uri))
        g.add((pop_uri, CTI.requiresBiomarker, bio_uri))

    # 3. SAFETY HIERARCHY (Study -> Report -> AE -> PT -> SOC)
    for safety in MOCK_SOURCES["Safety / RIM Content"]:
        study_uri = URIRef(f"https://w3id.org/cti/study/{safety['study_id']}")
        rep_uri = URIRef(f"https://w3id.org/cti/safetyreport/{safety['safety_report']['report_id']}")
        ae_uri = URIRef(f"https://w3id.org/cti/ae/{safety['safety_report']['adverse_event'].replace(' ', '')}")
        pt_uri = URIRef(f"https://w3id.org/cti/meddra_pt/{safety['safety_report']['meddra_pt'].replace(' ', '')}")
        soc_uri = URIRef(f"https://w3id.org/cti/meddra_soc/{safety['safety_report']['meddra_soc'].replace(' ', '')}")

        g.add((rep_uri, RDF.type, CTI.SafetyReport))
        g.add((rep_uri, RDFS.label, Literal(safety["safety_report"]["report_id"])))
        
        g.add((ae_uri, RDF.type, CTI.AdverseEvent))
        g.add((ae_uri, RDFS.label, Literal(safety["safety_report"]["adverse_event"])))
        
        g.add((pt_uri, RDF.type, CTI.MedDRAPT))
        g.add((pt_uri, RDFS.label, Literal(safety["safety_report"]["meddra_pt"])))
        
        g.add((soc_uri, RDF.type, CTI.MedDRASOC))
        g.add((soc_uri, RDFS.label, Literal(safety["safety_report"]["meddra_soc"])))

        g.add((study_uri, CTI.hasSafetyReport, rep_uri))
        g.add((rep_uri, CTI.reportsEvent, ae_uri))
        g.add((ae_uri, CTI.mappedToPT, pt_uri))
        g.add((pt_uri, CTI.belongsToSOC, soc_uri))

    # 4. SCIENTIFIC RESPONSE HIERARCHY (Study -> Document -> Topic -> Medical Concept)
    for sr in MOCK_SOURCES["Scientific Responses"]:
        study_uri = URIRef(f"https://w3id.org/cti/study/{sr['study_id']}")
        doc_uri = URIRef(f"https://w3id.org/cti/scires/{sr['response_doc']['doc_id']}")
        topic_uri = URIRef(f"https://w3id.org/cti/topic/{sr['response_doc']['inquiry_topic'].replace(' ', '')}")
        concept_uri = URIRef(f"https://w3id.org/cti/concept/{sr['response_doc']['medical_concept'].replace(' ', '')}")

        g.add((doc_uri, RDF.type, CTI.ScientificResponseDoc))
        g.add((doc_uri, RDFS.label, Literal(sr["response_doc"]["doc_id"])))
        
        g.add((topic_uri, RDF.type, CTI.InquiryTopic))
        g.add((topic_uri, RDFS.label, Literal(sr["response_doc"]["inquiry_topic"])))
        
        g.add((concept_uri, RDF.type, CTI.MedicalConcept))
        g.add((concept_uri, RDFS.label, Literal(sr["response_doc"]["medical_concept"])))

        g.add((study_uri, CTI.hasScientificResponse, doc_uri))
        g.add((doc_uri, CTI.addressesTopic, topic_uri))
        g.add((topic_uri, CTI.relatesToConcept, concept_uri))

    return g

kg = build_clinical_knowledge_graph()

# -------------------------------------------------------------------
# 3. DYNAMIC VIS.JS GRAPH VISUALIZATION
# -------------------------------------------------------------------
def generate_graph_html(kg_graph: Graph) -> str:
    """Dynamically parses the RDFLib graph to render all levels in Vis.js."""
    nodes = {}
    edges = []

    # Map ontological classes to specific UI colors
    color_map = {
        "ClinicalStudy": "#d51900",       # Red
        "DiseaseCondition": "#79529c",    # Purple
        "TherapeuticArea": "#ab47bc",
        "Treatment": "#1976d2",           # Blue
        "ActiveIngredient": "#0288d1",
        "MechanismOfAction": "#03a9f4",
        "Population": "#388e3c",          # Green
        "Biomarker": "#81c784",
        
        # New Safety Hierarchy Colors (Oranges/Reds)
        "SafetyReport": "#e65100",
        "AdverseEvent": "#f57c00",
        "MedDRAPT": "#ff9800",
        "MedDRASOC": "#ffb74d",
        
        # New Scientific Response Hierarchy Colors (Indigos)
        "ScientificResponseDoc": "#3949ab",
        "InquiryTopic": "#5c6bc0",
        "MedicalConcept": "#7986cb"
    }

    # Extract dynamic triples
    for s, p, o in kg_graph:
        if isinstance(o, URIRef): 
            for node_uri in [s, o]:
                node_id = str(node_uri)
                if node_id not in nodes:
                    node_type_uri = kg_graph.value(node_uri, RDF.type)
                    type_str = str(node_type_uri).split('#')[-1] if node_type_uri else "Node"
                    label = str(kg_graph.value(node_uri, RDFS.label) or node_id.split('/')[-1])
                    color = color_map.get(type_str, "#9e9e9e")

                    nodes[node_id] = {
                        "id": node_id,
                        "label": f"{type_str}\n{label}",
                        "color": color,
                        "shape": "box"
                    }

            edge_label = str(p).split('#')[-1]
            edges.append({
                "from": str(s),
                "to": str(o),
                "label": edge_label,
                "arrows": "to",
                "color": "#b0bec5"
            })

    node_list = list(nodes.values())
    
    raw_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
      <style>
        html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; }}
        #kg-network {{ width: 100%; height: 100%; background-color: #f8f9fa; }}
      </style>
    </head>
    <body>
      <div id="kg-network"></div>
      <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(node_list)});
        var edges = new vis.DataSet({json.dumps(edges)});
        var container = document.getElementById('kg-network');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
          nodes: {{ font: {{ color: '#ffffff', size: 11, face: 'Arial' }}, margin: 8, shadow: true }},
          edges: {{ font: {{ size: 10, align: 'middle', background: '#ffffff', color: '#607d8b' }}, smooth: {{ type: 'cubicBezier' }} }},
          layout: {{ hierarchical: {{ direction: 'UD', sortMethod: 'directed', nodeSpacing: 160, levelSeparation: 130 }} }},
          physics: {{ enabled: false }}
        }};
        new vis.Network(container, data, options);
      </script>
    </body>
    </html>
    """
    
    escaped_html = raw_html.replace('"', '&quot;')
    return f'<iframe srcdoc="{escaped_html}" style="width: 100%; height: 750px; border: 1px solid #e0e0e0; border-radius: 8px;"></iframe>'


# -------------------------------------------------------------------
# 4. LANGGRAPH ORCHESTRATOR
# -------------------------------------------------------------------
class GraphRAGState(TypedDict):
    query: str
    intent: str
    sparql_query: str
    final_response: str

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

def sparql_retriever_node(state: GraphRAGState) -> Dict[str, Any]:
    return {"sparql_query": "SELECT * WHERE { ?s ?p ?o } LIMIT 10"}

def synthesis_node(state: GraphRAGState) -> Dict[str, Any]:
    return {"final_response": f"Simulated Multi-Hop GraphRAG Response across Deep Safety and Evidence sources for: '{state['query']}'"}

workflow = StateGraph(GraphRAGState)
workflow.add_node("sparql_retriever", sparql_retriever_node)
workflow.add_node("synthesis", synthesis_node)
workflow.set_entry_point("sparql_retriever")
workflow.add_edge("sparql_retriever", "synthesis")
workflow.add_edge("synthesis", END)
orchestrator = workflow.compile()


# -------------------------------------------------------------------
# 5. GRADIO PORTAL UI
# -------------------------------------------------------------------
def search_pipeline(query: str):
    if not query.strip():
        return "", "", generate_graph_html(kg)
    result = orchestrator.invoke({"query": query, "intent": "", "sparql_query": "", "final_response": ""})
    return result["final_response"], result["sparql_query"], generate_graph_html(kg)

custom_css = "#main-title { font-size: 28px; font-weight: 700; color: #d51900; }"

with gr.Blocks(css=custom_css, title="iMedical Search") as demo:
    gr.HTML("<div id='main-title'>iMedical Search</div>")

    with gr.Row():
        search_box = gr.Textbox(placeholder="Search across trials, safety signals, PK concepts...", scale=9)
        search_btn = gr.Button("Search", variant="primary", scale=1)

    with gr.Tabs():
        with gr.TabItem("Knowledge Graph Hierarchy (Multi-Level)"):
            gr.Markdown("#### 🕸️ Expanded Evidence Graph (Safety, Pharmacokinetics, Operations)")
            kg_html_display = gr.HTML(value=generate_graph_html(kg))
            
        with gr.TabItem("Connected Intelligence Answer"):
            answer_display = gr.Markdown()
            sparql_display = gr.Code(language="sql")

    search_btn.click(fn=search_pipeline, inputs=[search_box], outputs=[answer_display, sparql_display, kg_html_display])
    search_box.submit(fn=search_pipeline, inputs=[search_box], outputs=[answer_display, sparql_display, kg_html_display])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)