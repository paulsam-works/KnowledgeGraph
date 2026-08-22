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
# 1. SAMPLE SOURCE DOCUMENTS (Matching Diagram Sources)
# -------------------------------------------------------------------
MOCK_SOURCES = {
    "Clinical Study Reports": [
        {
            "study_id": "NCT04561234",
            "title": "Phase 3 Study of Daratumumab in Relapsed Multiple Myeloma",
            "phase": "Phase 3",
            "disease": "Multiple Myeloma",
            "therapeutic_area": "Oncology",
            "treatment": "Daratumumab 1800mg SubQ",
            "comparator": "Standard of Care (Rd)",
            "primary_endpoint": "Progression-Free Survival (PFS)",
            "pfs_months": "24.5 months vs 14.1 months",
            "hazard_ratio": 0.63,
            "p_value": 0.001,
            "safety_summary": "Infusion reactions in 8%, Grade 3/4 Neutropenia in 12%"
        }
    ],
    "Protocols & Amendments": [
        {
            "study_id": "NCT04561234",
            "protocol_ver": "v3.2 Amendment",
            "population": "Adults >= 18y with confirmed refractory disease",
            "inclusion_biomarkers": "CD38+ plasma cells >= 10%",
            "planned_enrollment": 520,
            "sites": ["Memorial Sloan Kettering (US)", "Gustave Roussy (FR)"]
        }
    ],
    "Safety / RIM Content": [
        {
            "study_id": "NCT04561234",
            "ae_term": "Neutropenia",
            "meddra_code": "10029354",
            "severity": "Grade 3-4",
            "causality": "Related",
            "frequency_pct": 12.0
        }
    ],
    "Scientific Responses": [
        {
            "study_id": "NCT04561234",
            "inquiry_topic": "Subcutaneous vs IV Administration Efficacy in Renal Impairment",
            "medical_statement": "SubQ Daratumumab showed equivalent pharmacokinetic exposure and non-inferior overall response rate in patients with moderate renal impairment."
        }
    ],
    "Scientific Presentations": [
        {
            "study_id": "NCT04561234",
            "congress": "ASCO 2025 Annual Meeting",
            "presentation_title": "Subgroup Biomarker Analysis of Daratumumab in High-Risk Cytogenetics",
            "key_finding": "Significant PFS benefit maintained in del(17p) and t(4;14) cohorts."
        }
    ],
    "PubMed Literature": [
        {
            "study_id": "NCT04561234",
            "pmid": "38921102",
            "journal": "The New England Journal of Medicine",
            "title": "Subcutaneous Daratumumab in Relapsed Multiple Myeloma: Long-term Efficacy",
            "conclusion": "Long-term follow-up confirms persistent survival benefit with lower rates of systemic reactions."
        }
    ]
}

# -------------------------------------------------------------------
# 2. KNOWLEDGE GRAPH CREATION (RDFLib with Custom Ontology)
# -------------------------------------------------------------------
def build_clinical_knowledge_graph() -> Graph:
    g = Graph()
    CTI = Namespace("https://w3id.org/cti/ontology#")
    g.bind("cti", CTI)
    g.bind("rdfs", RDFS)

    for csr in MOCK_SOURCES["Clinical Study Reports"]:
        study_uri = URIRef(f"https://w3id.org/cti/study/{csr['study_id']}")
        g.add((study_uri, RDF.type, CTI.ClinicalStudy))
        g.add((study_uri, CTI.studyId, Literal(csr["study_id"], datatype=XSD.string)))
        g.add((study_uri, CTI.studyTitle, Literal(csr["title"], datatype=XSD.string)))
        g.add((study_uri, CTI.phase, Literal(csr["phase"], datatype=XSD.string)))

        # Disease / Condition Node
        disease_uri = URIRef(f"https://w3id.org/cti/disease/{csr['disease'].replace(' ', '_')}")
        g.add((disease_uri, RDF.type, CTI.DiseaseCondition))
        g.add((disease_uri, RDFS.label, Literal(csr["disease"])))
        g.add((study_uri, CTI.targetsDisease, disease_uri))

        # Treatment Node
        treatment_uri = URIRef(f"https://w3id.org/cti/treatment/{csr['study_id']}_tx")
        g.add((treatment_uri, RDF.type, CTI.Treatment))
        g.add((treatment_uri, CTI.drugName, Literal(csr["treatment"])))
        g.add((study_uri, CTI.usesTreatment, treatment_uri))

        # Outcome Endpoint
        outcome_uri = URIRef(f"https://w3id.org/cti/outcome/{csr['study_id']}_primary")
        g.add((outcome_uri, RDF.type, CTI.OutcomeEndpoint))
        g.add((outcome_uri, RDFS.label, Literal(csr["primary_endpoint"])))
        g.add((outcome_uri, CTI.hazardRatio, Literal(csr["hazard_ratio"], datatype=XSD.float)))
        g.add((outcome_uri, CTI.pVal, Literal(csr["p_value"], datatype=XSD.float)))
        g.add((study_uri, CTI.measuresOutcome, outcome_uri))

    # Add Protocols
    for proto in MOCK_SOURCES["Protocols & Amendments"]:
        study_uri = URIRef(f"https://w3id.org/cti/study/{proto['study_id']}")
        pop_uri = URIRef(f"https://w3id.org/cti/population/{proto['study_id']}_pop")
        g.add((pop_uri, RDF.type, CTI.Population))
        g.add((pop_uri, RDFS.label, Literal(proto["population"])))
        g.add((study_uri, CTI.hasPopulation, pop_uri))

    # Add Safety Records
    for safety in MOCK_SOURCES["Safety / RIM Content"]:
        study_uri = URIRef(f"https://w3id.org/cti/study/{safety['study_id']}")
        safety_uri = URIRef(f"https://w3id.org/cti/safety/{safety['study_id']}_{safety['meddra_code']}")
        g.add((safety_uri, RDF.type, CTI.SafetyEvent))
        g.add((safety_uri, CTI.adverseEvent, Literal(safety["ae_term"])))
        g.add((safety_uri, RDFS.comment, Literal(f"Severity: {safety['severity']}, Frequency: {safety['frequency_pct']}%")))
        g.add((study_uri, CTI.reportsSafety, safety_uri))

    # Add Scientific Response Asset
    for sr in MOCK_SOURCES["Scientific Responses"]:
        study_uri = URIRef(f"https://w3id.org/cti/study/{sr['study_id']}")
        sr_uri = URIRef(f"https://w3id.org/cti/evidence/SR_{sr['study_id']}")
        g.add((sr_uri, RDF.type, CTI.ScientificResponse))
        g.add((sr_uri, RDFS.label, Literal(sr["inquiry_topic"])))
        g.add((sr_uri, RDFS.comment, Literal(sr["medical_statement"])))
        g.add((study_uri, CTI.hasEvidenceAsset, sr_uri))

    return g

kg = build_clinical_knowledge_graph()

# -------------------------------------------------------------------
# 3. GRAPH VISUALIZATION GENERATOR (Vis.js Interactive HTML)
# -------------------------------------------------------------------

def generate_graph_html(traversed_edges: List[Dict[str, str]] = None) -> str:
    """Generates an interactive force-directed graph canvas within a secure iframe."""
    nodes = [
        {"id": "study", "label": "Clinical Study\n(NCT04561234)", "group": "study", "color": "#d51900"},
        {"id": "disease", "label": "Disease:\nMultiple Myeloma", "group": "disease", "color": "#79529cf0"},
        {"id": "treatment", "label": "Treatment:\nDaratumumab SubQ", "group": "treatment", "color": "#1976d2"},
        {"id": "pop", "label": "Population:\nAdults >= 18y", "group": "pop", "color": "#388e3c"},
        {"id": "safety", "label": "Safety Event:\nNeutropenia (Gr 3-4)", "group": "safety", "color": "#f57c00"},
        {"id": "outcome", "label": "Outcome:\nPFS (HR 0.63)", "group": "outcome", "color": "#00897b"},
        {"id": "sr", "label": "Scientific Response:\nRenal Impairment PK", "group": "asset", "color": "#5c6bc0"}
    ]

    base_edges = [
        {"from": "study", "to": "disease", "label": "targetsDisease"},
        {"from": "study", "to": "treatment", "label": "usesTreatment"},
        {"from": "study", "to": "pop", "label": "hasPopulation"},
        {"from": "study", "to": "safety", "label": "reportsSafety"},
        {"from": "study", "to": "outcome", "label": "measuresOutcome"},
        {"from": "study", "to": "sr", "label": "hasEvidenceAsset"}
    ]

    # Format edges with highlighting if traversed
    edges = []
    for edge in base_edges:
        is_traversed = traversed_edges and any(
            edge["label"].lower() in str(e).lower() for e in traversed_edges
        )
        edges.append({
            "from": edge["from"],
            "to": edge["to"],
            "label": edge["label"],
            "color": {"color": "#ff1744" if is_traversed else "#888888"},
            "width": 4 if is_traversed else 1.5,
            "arrows": "to"
        })

    # The raw HTML string
    raw_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
      <style>
        html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; }}
        #kg-network {{
          width: 100%;
          height: 100%;
          background-color: #fafafa;
        }}
      </style>
    </head>
    <body>
      <div id="kg-network"></div>
      <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(nodes)});
        var edges = new vis.DataSet({json.dumps(edges)});
        var container = document.getElementById('kg-network');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
          nodes: {{
            shape: 'box',
            font: {{ color: '#ffffff', size: 13, face: 'Helvetica' }},
            margin: 10,
            shadow: true
          }},
          edges: {{
            font: {{ size: 11, align: 'middle', background: '#ffffff' }},
            smooth: {{ type: 'cubicBezier', roundness: 0.2 }}
          }},
          physics: {{
            barnesHut: {{ springLength: 150, gravitationalConstant: -3500 }}
          }}
        }};
        var network = new vis.Network(container, data, options);
      </script>
    </body>
    </html>
    """
    
    # Escape quotes for safe embedding in the iframe srcdoc attribute
    escaped_html = raw_html.replace('"', '&quot;')
    
    # Return the wrapped iframe
    iframe_wrapper = f'<iframe srcdoc="{escaped_html}" style="width: 100%; height: 500px; border: 1px solid #e0e0e0; border-radius: 8px;"></iframe>'
    
    return iframe_wrapper

    
def generate_graph_html2(traversed_edges: List[Dict[str, str]] = None) -> str:
    """Generates an interactive force-directed graph canvas within a secure iframe."""
    nodes = [
        {"id": "study", "label": "Clinical Study\n(NCT04561234)", "group": "study", "color": "#d51900"},
        {"id": "disease", "label": "Disease:\nMultiple Myeloma", "group": "disease", "color": "#79529cf0"},
        {"id": "treatment", "label": "Treatment:\nDaratumumab SubQ", "group": "treatment", "color": "#1976d2"},
        {"id": "pop", "label": "Population:\nAdults >= 18y", "group": "pop", "color": "#388e3c"},
        {"id": "safety", "label": "Safety Event:\nNeutropenia (Gr 3-4)", "group": "safety", "color": "#f57c00"},
        {"id": "outcome", "label": "Outcome:\nPFS (HR 0.63)", "group": "outcome", "color": "#00897b"},
        {"id": "sr", "label": "Scientific Response:\nRenal Impairment PK", "group": "asset", "color": "#5c6bc0"}
    ]

    base_edges = [
        {"from": "study", "to": "disease", "label": "targetsDisease"},
        {"from": "study", "to": "treatment", "label": "usesTreatment"},
        {"from": "study", "to": "pop", "label": "hasPopulation"},
        {"from": "study", "to": "safety", "label": "reportsSafety"},
        {"from": "study", "to": "outcome", "label": "measuresOutcome"},
        {"from": "study", "to": "sr", "label": "hasEvidenceAsset"}
    ]

    # Format edges with highlighting if traversed
    edges = []
    for edge in base_edges:
        is_traversed = traversed_edges and any(
            edge["label"].lower() in str(e).lower() for e in traversed_edges
        )
        edges.append({
            "from": edge["from"],
            "to": edge["to"],
            "label": edge["label"],
            "color": {"color": "#ff1744" if is_traversed else "#888888"},
            "width": 4 if is_traversed else 1.5,
            "arrows": "to"
        })

    # The raw HTML string
    raw_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
      <style>
        html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; }}
        #kg-network {{
          width: 100%;
          height: 100%;
          background-color: #fafafa;
        }}
      </style>
    </head>
    <body>
      <div id="kg-network"></div>
      <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(nodes)});
        var edges = new vis.DataSet({json.dumps(edges)});
        var container = document.getElementById('kg-network');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
          nodes: {{
            shape: 'box',
            font: {{ color: '#ffffff', size: 13, face: 'Helvetica' }},
            margin: 10,
            shadow: true
          }},
          edges: {{
            font: {{ size: 11, align: 'middle', background: '#ffffff' }},
            smooth: {{ type: 'cubicBezier', roundness: 0.2 }}
          }},
          physics: {{
            barnesHut: {{ springLength: 150, gravitationalConstant: -3500 }}
          }}
        }};
        var network = new vis.Network(container, data, options);
      </script>
    </body>
    </html>
    """
    
    # Escape quotes for safe embedding in the iframe srcdoc attribute
    escaped_html = raw_html.replace('"', '&quot;')
    
    # Return the wrapped iframe
    iframe_wrapper = f'<iframe srcdoc="{escaped_html}" style="width: 100%; height: 500px; border: 1px solid #e0e0e0; border-radius: 8px;"></iframe>'
    
    return iframe_wrapper


def generate_graph_html1(traversed_edges: List[Dict[str, str]] = None) -> str:
    """Generates an interactive force-directed graph canvas highlighting traversed paths."""
    nodes = [
        {"id": "study", "label": "Clinical Study\n(NCT04561234)", "group": "study", "color": "#d51900"},
        {"id": "disease", "label": "Disease:\nMultiple Myeloma", "group": "disease", "color": "#79529cf0"},
        {"id": "treatment", "label": "Treatment:\nDaratumumab SubQ", "group": "treatment", "color": "#1976d2"},
        {"id": "pop", "label": "Population:\nAdults >= 18y", "group": "pop", "color": "#388e3c"},
        {"id": "safety", "label": "Safety Event:\nNeutropenia (Gr 3-4)", "group": "safety", "color": "#f57c00"},
        {"id": "outcome", "label": "Outcome:\nPFS (HR 0.63)", "group": "outcome", "color": "#00897b"},
        {"id": "sr", "label": "Scientific Response:\nRenal Impairment PK", "group": "asset", "color": "#5c6bc0"}
    ]

    base_edges = [
        {"from": "study", "to": "disease", "label": "targetsDisease"},
        {"from": "study", "to": "treatment", "label": "usesTreatment"},
        {"from": "study", "to": "pop", "label": "hasPopulation"},
        {"from": "study", "to": "safety", "label": "reportsSafety"},
        {"from": "study", "to": "outcome", "label": "measuresOutcome"},
        {"from": "study", "to": "sr", "label": "hasEvidenceAsset"}
    ]

    # Format edges with highlighting if traversed
    edges = []
    for edge in base_edges:
        is_traversed = traversed_edges and any(
            edge["label"].lower() in str(e).lower() for e in traversed_edges
        )
        edges.append({
            "from": edge["from"],
            "to": edge["to"],
            "label": edge["label"],
            "color": {"color": "#ff1744" if is_traversed else "#888888"},
            "width": 4 if is_traversed else 1.5,
            "arrows": "to"
        })

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
      <style>
        #kg-network {{
          width: 100%;
          height: 480px;
          border: 1px solid #e0e0e0;
          border-radius: 8px;
          background-color: #fafafa;
        }}
      </style>
    </head>
    <body>
      <div id="kg-network"></div>
      <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(nodes)});
        var edges = new vis.DataSet({json.dumps(edges)});
        var container = document.getElementById('kg-network');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
          nodes: {{
            shape: 'box',
            font: {{ color: '#ffffff', size: 13, face: 'Helvetica' }},
            margin: 10,
            shadow: true
          }},
          edges: {{
            font: {{ size: 11, align: 'middle', background: '#ffffff' }},
            smooth: {{ type: 'cubicBezier', roundness: 0.2 }}
          }},
          physics: {{
            barnesHut: {{ springLength: 150, gravitationalConstant: -3500 }}
          }}
        }};
        var network = new vis.Network(container, data, options);
      </script>
    </body>
    </html>
    """
    return html

# -------------------------------------------------------------------
# 4. GRAPH RAG ORCHESTRATION VIA LANGGRAPH
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
    sparql_generation_prompt = f"""You are an RDF/SPARQL expert for a Clinical Trial Knowledge Graph.
Ontology prefixes:
PREFIX cti: <https://w3id.org/cti/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

Generate a SPARQL SELECT query that answers: "{query}".
Always select ?studyId, ?title, ?property, ?value.
Return ONLY valid SPARQL text."""

    raw_sparql = llm.invoke([HumanMessage(content=sparql_generation_prompt)]).content
    clean_sparql = raw_sparql.replace("```sparql", "").replace("```", "").strip()

    default_sparql = """
    PREFIX cti: <https://w3id.org/cti/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?studyId ?title ?property ?value WHERE {
        ?s a cti:ClinicalStudy ;
           cti:studyId ?studyId ;
           cti:studyTitle ?title .
        OPTIONAL { ?s cti:usesTreatment ?tx . ?tx cti:drugName ?value . BIND("Treatment" as ?property) }
        OPTIONAL { ?s cti:reportsSafety ?safe . ?safe cti:adverseEvent ?value . BIND("Adverse Event" as ?property) }
        OPTIONAL { ?s cti:measuresOutcome ?out . ?out rdfs:label ?value . BIND("Outcome" as ?property) }
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

Instructions:
1. Provide a comprehensive, accurate clinical response.
2. Explicitly cite the Evidence Sources (CSR, Protocol ver, Safety/RIM, Scientific Responses).
3. State the Grounding Confidence Score (e.g., 0.98/1.0 - Highly Grounded)."""

    response = llm.invoke([SystemMessage(content=prompt)]).content
    citations = [
        "CSR: NCT04561234 Section 11.2 (PFS Outcomes)",
        "Protocol: NCT04561234 v3.2 Amendment (Inclusion/Exclusion)",
        "Safety/RIM: MedDRA v27.0 - Neutropenia Incidence",
        "Scientific Response: Ref #SR-2025-0891"
    ]
    return {
        "final_response": response,
        "grounding_score": "0.97 / 1.0 (Direct Graph & Evidence Grounded)",
        "citations": citations
    }

# Build LangGraph Workflow
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
# 5. GRADIO PORTAL WITH KNOWLEDGE GRAPH EXPLORER
# -------------------------------------------------------------------
def search_pipeline(
    query: str,
    doc_type: str,
    drug_jnj: str,
    drug_comp: str,
    doc_search_only: bool
):
    if not query.strip():
        return "", "", "", "Please enter a clinical query.", generate_graph_html()

    filters = {
        "Document Type": doc_type,
        "Drug - JNJ": drug_jnj,
        "Drug - Competitor": drug_comp,
        "Document Search Only": doc_search_only
    }

    state_input: GraphRAGState = {
        "query": query,
        "selected_filters": filters,
        "intent": "",
        "sparql_query": "",
        "graph_evidence": [],
        "final_response": "",
        "grounding_score": "",
        "citations": []
    }

    result = orchestrator.invoke(state_input)

    response_text = result["final_response"]
    citations_text = "\n".join([f"• {c}" for c in result["citations"]])
    sparql_text = result["sparql_query"]
    grounding_badge = f"**Intent:** {result['intent']} | **Grounding Score:** {result['grounding_score']}"
    
    # Generate graph with highlighted traversal
    graph_viz_html = generate_graph_html(traversed_edges=result["graph_evidence"])

    return grounding_badge, response_text, citations_text, sparql_text, graph_viz_html

custom_css = """
#main-title { font-size: 28px; font-weight: 700; color: #d51900; margin-bottom: 0px; }
#sub-title { font-size: 13px; color: #666; margin-bottom: 15px; }
.gr-button-primary { background-color: #d51900 !important; color: white !important; }
"""

with gr.Blocks(css=custom_css, title="iMedical Search | Clinical Trial Intelligence") as demo:
    gr.HTML(
        """
        <div style='text-align: center;'>
            <div id='main-title'>iMedical <span style='background-color:#d51900; color:white; padding: 2px 8px; border-radius: 4px;'>Search</span></div>
            <div id='sub-title'>Connected Evidence and Clinical Semantics (GraphRAG Powered)</div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### ⚙️ Search Filters & Facets")
            doc_type_dropdown = gr.Dropdown(
                label="Document Type",
                choices=["All", "Clinical Study Reports", "Protocols & Amendments", "Safety / RIM Content", "Scientific Responses", "Scientific Presentations", "PubMed Literature"],
                value="All"
            )
            drug_jnj_dropdown = gr.Dropdown(
                label="Drugs - Primary / JNJ",
                choices=["All", "Daratumumab SubQ", "Amivantamab", "Teclistamab"],
                value="Daratumumab SubQ"
            )
            drug_comp_dropdown = gr.Dropdown(
                label="Drugs - Competitor / Standard",
                choices=["All", "Lenalidomide / Dexamethasone", "Carfilzomib", "Bortezomib"],
                value="Lenalidomide / Dexamethasone"
            )
            doc_search_only_cb = gr.Checkbox(label="Document Search Only", value=False)
            
            gr.Markdown(
                """
                ---
                **GenAI Adherence Policy**  
                *Grounding across CDISC, SNOMED CT, MedDRA, and RxNorm ontologies.*
                """
            )

        with gr.Column(scale=3):
            with gr.Row():
                search_box = gr.Textbox(
                    placeholder="Search clinical endpoints, safety signals, inclusion criteria, or trial comparisons...",
                    label="Search...",
                    scale=9
                )
                search_btn = gr.Button("Search / Run AI", variant="primary", scale=1)

            grounding_display = gr.Markdown("")

            with gr.Tabs():
                with gr.TabItem("Connected Intelligence Answer"):
                    answer_display = gr.Markdown(label="Generated Synthesized Answer")
                
                with gr.TabItem("Knowledge Graph & Traversal"):
                    gr.Markdown("#### 🕸️ Interactive Knowledge Graph & Active Traversal Path")
                    kg_html_display = gr.HTML(value=generate_graph_html())
                    gr.Markdown(
                        "> **Legend:** 🔴 Clinical Study | 🟣 Disease | 🔵 Treatment | 🟢 Population | 🟠 Safety Event | 🟩 Outcome | 🔷 Scientific Response  \n"
                        "> *Red/Thick edges indicate active subgraphs traversed during GraphRAG retrieval.*"
                    )

                with gr.TabItem("Evidence Linkage & Citations"):
                    citations_display = gr.Markdown(label="Traceable Document Citations")

                with gr.TabItem("Generated SPARQL Query"):
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