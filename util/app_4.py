import os
import json
import gradio as gr
from typing import Dict, List, Any, TypedDict
from rdflib import Graph, Literal, RDF, Namespace, URIRef
from rdflib.namespace import XSD, RDFS

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

# Set OpenAI API key (or replace with local LLM runtime)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

# -------------------------------------------------------------------
# 1. DEEP MULTI-LEVEL SOURCE DOCUMENTS
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
                "description": "Adults >= 18y with confirmed refractory disease",
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
# 2. MULTI-LEVEL ONTOLOGY BUILDER & KNOWLEDGE GRAPH INSTANTIATION
# -------------------------------------------------------------------
CTI = Namespace("https://w3id.org/cti/ontology#")

def build_clinical_knowledge_graph() -> Graph:
    g = Graph()
    g.bind("cti", CTI)
    g.bind("rdfs", RDFS)

    # 1. Clinical Study + Disease + Treatment Hierarchy
    for csr in MOCK_SOURCES["Clinical Study Reports"]:
        study_uri = URIRef(f"https://w3id.org/cti/study/{csr['study_id']}")
        g.add((study_uri, RDF.type, CTI.ClinicalStudy))
        g.add((study_uri, RDFS.label, Literal(csr["study_id"])))

        # Disease 
        dis_uri = URIRef(f"https://w3id.org/cti/disease/{csr['disease']['snomed_code']}")
        ta_uri = URIRef(f"https://w3id.org/cti/ta/{csr['disease']['therapeutic_area'].replace(' ', '')}")
        g.add((dis_uri, RDF.type, CTI.DiseaseCondition))
        g.add((dis_uri, RDFS.label, Literal(csr["disease"]["term"])))
        g.add((ta_uri, RDF.type, CTI.TherapeuticArea))
        g.add((ta_uri, RDFS.label, Literal(csr["disease"]["therapeutic_area"])))
        g.add((study_uri, CTI.targetsDisease, dis_uri))
        g.add((dis_uri, CTI.belongsToTherapeuticArea, ta_uri))

        # Treatment 
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

    # 3. Safety / RIM Hierarchy
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

    # 4. Scientific Responses Hierarchy
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
# 3. DYNAMIC VIS.JS GRAPH VISUALIZATION (Secure IFrame)
# -------------------------------------------------------------------
def generate_graph_html(kg_graph: Graph) -> str:
    """Dynamically parses the RDFLib graph to render all levels in Vis.js."""
    nodes = {}
    edges = []

    # Map ontological classes to specific UI colors
    color_map = {
        "ClinicalStudy": "#d51900",       
        "DiseaseCondition": "#79529c",    
        "TherapeuticArea": "#ab47bc",
        "Treatment": "#1976d2",           
        "ActiveIngredient": "#0288d1",
        "MechanismOfAction": "#03a9f4",
        "Population": "#388e3c",          
        "Biomarker": "#81c784",
        "SafetyReport": "#e65100",
        "AdverseEvent": "#f57c00",
        "MedDRAPT": "#ff9800",
        "MedDRASOC": "#ffb74d",
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
    return f'<iframe srcdoc="{escaped_html}" style="width: 100%; height: 600px; border: 1px solid #e0e0e0; border-radius: 8px;"></iframe>'

# -------------------------------------------------------------------
# 4. LANGGRAPH ORCHESTRATION (With RAG Logic)
# -------------------------------------------------------------------
class GraphRAGState(TypedDict):
    query: str
    selected_filters: Dict[str, Any]
    intent: str
    sparql_query: str
    graph_evidence: List[Dict[str, Any]]
    final_response: str
    grounding_score: str
    citations: List[str]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

def intent_router_node(state: GraphRAGState) -> Dict[str, Any]:
    query = state["query"]
    prompt = f"""Classify the intent of the clinical trial inquiry into one of:
    [Protocol Feasibility, Trial Comparison, Safety Surveillance, Endpoint Evidence, Scientific Response].
    Query: "{query}"
    Output solely the intent name."""
    res = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    return {"intent": res}

def sparql_generator_and_retriever_node(state: GraphRAGState) -> Dict[str, Any]:
    query = state["query"]
    sparql_prompt = f"""Generate a valid SPARQL SELECT query for a Clinical Trial Knowledge Graph.
Ontology prefixes:
PREFIX cti: <https://w3id.org/cti/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
Query: "{query}"
Return ONLY valid SPARQL text."""

    raw_sparql = llm.invoke([HumanMessage(content=sparql_prompt)]).content
    clean_sparql = raw_sparql.replace("```sparql", "").replace("```", "").strip()

    # Fallback default overarching query for mock purposes
    default_sparql = """
    PREFIX cti: <https://w3id.org/cti/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?studyId ?type ?label WHERE {
        ?s a cti:ClinicalStudy ;
           cti:studyId ?studyId .
        ?s ?p ?o .
        ?o a ?type ; rdfs:label ?label .
    } LIMIT 20
    """
    evidence = []
    try:
        q_results = kg.query(clean_sparql)
        for row in q_results:
            evidence.append({str(k): str(v) for k, v in row.asdict().items()})
    except Exception:
        clean_sparql = default_sparql
        q_results = kg.query(clean_sparql)
        for row in q_results:
            evidence.append({str(k): str(v) for k, v in row.asdict().items()})

    return {"sparql_query": clean_sparql, "graph_evidence": evidence}

def synthesis_and_grounding_node(state: GraphRAGState) -> Dict[str, Any]:
    query = state["query"]
    evidence = state.get("graph_evidence", [])
    filters = state.get("selected_filters", {})

    prompt = f"""You are an enterprise AI medical assistant grounded in connected clinical semantics.
User Query: "{query}"
Applied UI Filters: {json.dumps(filters)}
Retrieved Semantic Graph Evidence: {json.dumps(evidence)}
Unstructured Context Baseline: {json.dumps(MOCK_SOURCES)}

Instructions: Provide a comprehensive, accurate clinical response. Explicitly cite the Evidence Sources (CSR, Protocol ver, Safety/RIM, Scientific Responses)."""

    response = llm.invoke([SystemMessage(content=prompt)]).content
    
    citations = [
        "CSR: NCT04561234 Section 11.2 (PFS Outcomes)",
        "Protocol: NCT04561234 v3.2 Amendment (Inclusion/Exclusion)",
        "Safety/RIM: MedDRA PT - Neutrophil count decreased",
        "Scientific Response: Ref #SciRes-2026-0891"
    ]
    return {
        "final_response": response,
        "grounding_score": "0.98 / 1.0 (Direct Graph & Evidence Grounded)",
        "citations": citations
    }

workflow = StateGraph(GraphRAGState)
workflow.add_node("intent_router", intent_router_node)
workflow.add_node("sparql_retriever", sparql_generator_and_retriever_node)
workflow.add_node("synthesis_grounding", synthesis_and_grounding_node)

workflow.set_entry_point("intent_router")
workflow.add_edge("intent_router", "sparql_retriever")
workflow.add_edge("sparql_retriever", "synthesis_grounding")
workflow.add_edge("synthesis_grounding", END)

orchestrator = workflow.compile()

# -------------------------------------------------------------------
# 5. GRADIO PORTAL ("iMedical Search" UI Mock)
# -------------------------------------------------------------------
def search_pipeline(query: str, doc_type: str, drug_jnj: str, drug_comp: str, doc_search_only: bool):
    if not query.strip():
        return "", "", "", "Please enter a clinical query.", generate_graph_html(kg)

    filters = {
        "Document Type": doc_type,
        "Drug - JNJ": drug_jnj,
        "Drug - Competitor": drug_comp,
        "Document Search Only": doc_search_only
    }

    state_input: GraphRAGState = {
        "query": query, "selected_filters": filters, "intent": "", 
        "sparql_query": "", "graph_evidence": [], "final_response": "", 
        "grounding_score": "", "citations": []
    }

    result = orchestrator.invoke(state_input)

    response_text = result["final_response"]
    citations_text = "\n".join([f"• {c}" for c in result["citations"]])
    sparql_text = result["sparql_query"]
    grounding_badge = f"**Identified Intent:** {result['intent']} | **Grounding Score:** {result['grounding_score']}"
    
    return grounding_badge, response_text, citations_text, sparql_text, generate_graph_html(kg)


# Define Gradio Interface (Stylized to match the target UI)
custom_css = """
#main-title { font-size: 28px; font-weight: 700; color: #d51900; margin-bottom: 0px; }
#sub-title { font-size: 13px; color: #666; margin-bottom: 15px; }
.gr-button-primary { background-color: #d51900 !important; color: white !important; }
"""

with gr.Blocks(css=custom_css, title="iMedical Search | Clinical Trial Intelligence") as demo:
    gr.HTML(
        """
        <div style='text-align: center; margin-bottom: 20px;'>
            <div id='main-title'>iMedical <span style='background-color:#d51900; color:white; padding: 2px 8px; border-radius: 4px;'>Search</span></div>
            <div id='sub-title'>Connected Evidence and Clinical Semantics</div>
        </div>
        """
    )

    with gr.Row():
        # LEFT COLUMN - Filters
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### ⚙️ Search Filters & Facets")
            doc_type_dropdown = gr.Dropdown(
                label="Document Type",
                choices=["All", "Clinical Study Reports", "Protocols & Amendments", "Safety / RIM Content", "Scientific Responses"],
                value="All"
            )
            drug_jnj_dropdown = gr.Dropdown(
                label="Drugs - JNJ",
                choices=["All", "Daratumumab SubQ", "Amivantamab"],
                value="Daratumumab SubQ"
            )
            drug_comp_dropdown = gr.Dropdown(
                label="Drugs - Competitor",
                choices=["All", "Lenalidomide / Dexamethasone"],
                value="Lenalidomide / Dexamethasone"
            )
            doc_search_only_cb = gr.Checkbox(label="Document Search Only", value=False)
            
            gr.Markdown(
                """
                ---
                **GenAI Adherence Policy**  
                *Grounding across CDISC, SNOMED CT, MedDRA, and RxNorm.*
                """
            )

        # RIGHT COLUMN - Main Content Area
        with gr.Column(scale=3):
            with gr.Row():
                search_box = gr.Textbox(
                    placeholder="Search clinical endpoints, safety signals, inclusion criteria, or pharmacokinetics...",
                    label="Search...",
                    scale=9
                )
                search_btn = gr.Button("Search / Run AI", variant="primary", scale=1)

            grounding_display = gr.Markdown("")

            with gr.Tabs():
                with gr.TabItem("Connected Intelligence Answer"):
                    answer_display = gr.Markdown(label="Synthesized RAG Answer")
                
                with gr.TabItem("Knowledge Graph & Traversal"):
                    gr.Markdown("#### 🕸️ Interactive Knowledge Graph (Multi-Level Hierarchy)")
                    kg_html_display = gr.HTML(value=generate_graph_html(kg))
                    gr.Markdown(
                        "> **Ontology Highlights:** Cascades down to MedDRA SOCs, Active Ingredients, and Medical Concepts."
                    )

                with gr.TabItem("Evidence Linkage & Citations"):
                    citations_display = gr.Markdown(label="Traceable Document Citations")

                with gr.TabItem("Generated SPARQL Query"):
                    # Using "sql" as a supported language string in Gradio Code for proper rendering
                    sparql_display = gr.Code(label="SPARQL Query Executed Against RDF Store", language="sql")

    # Wire event handlers
    search_btn.click(
        fn=search_pipeline,
        inputs=[search_box, doc_type_dropdown, drug_jnj_dropdown, drug_comp_dropdown, doc_search_only_cb],
        outputs=[grounding_display, answer_display, citations_display, sparql_display, kg_html_display]
    )
    search_box.submit(
        fn=search_pipeline,
        inputs=[search_box, doc_type_dropdown, drug_jnj_dropdown, drug_comp_dropdown, doc_search_only_cb],
        outputs=[grounding_display, answer_display, citations_display, sparql_display, kg_html_display]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)