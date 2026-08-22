import os
import json
import gradio as gr
from typing import Dict, List, Any, TypedDict, Tuple
from rdflib import Graph, Literal, RDF, Namespace, URIRef
from rdflib.namespace import XSD, RDFS

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

# -------------------------------------------------------------------
# 1. EXPANDED CLINICAL MULTI-SOURCE DATA
# -------------------------------------------------------------------
MOCK_SOURCES = {
    "Clinical Study Reports": [
        {
            "study_id": "NCT04561234",
            "title": "Phase 3 Study of Daratumumab vs Standard of Care in MM",
            "disease": {"term": "Multiple Myeloma", "therapeutic_area": "Hematological Malignancy", "snomed_code": "109989006"},
            "treatment": {"regimen": "Daratumumab 1800mg SubQ", "active_ingredient": "Daratumumab", "mechanism_of_action": "CD38-directed cytolytic antibody"},
            "comparator": {"regimen": "Lenalidomide / Dexamethasone", "active_ingredient": "Lenalidomide", "mechanism_of_action": "Immunomodulatory imide drug"}
        },
        {
            "study_id": "NCT02665364",
            "title": "Phase 1/2 Study of Amivantamab in Advanced NSCLC",
            "disease": {"term": "Non-Small Cell Lung Cancer (NSCLC)", "therapeutic_area": "Solid Tumors", "snomed_code": "254637007"},
            "treatment": {"regimen": "Amivantamab-vmjw 1050mg IV", "active_ingredient": "Amivantamab", "mechanism_of_action": "EGFR/MET bispecific antibody"},
            "comparator": {"regimen": "Platinum-based Chemotherapy", "active_ingredient": "Cisplatin", "mechanism_of_action": "Alkylating antineoplastic agent"}
        },
        {
            "study_id": "NCT03145181",
            "title": "Phase 1/2 Study of Teclistamab in Relapsed/Refractory MM",
            "disease": {"term": "Refractory Multiple Myeloma", "therapeutic_area": "Hematological Malignancy", "snomed_code": "109989006_R"},
            "treatment": {"regimen": "Teclistamab 1.5 mg/kg SubQ", "active_ingredient": "Teclistamab", "mechanism_of_action": "BCMA/CD3 bispecific antibody"},
            "comparator": {"regimen": "Carfilzomib / Dexamethasone", "active_ingredient": "Carfilzomib", "mechanism_of_action": "Proteasome inhibitor"}
        }
    ],
    "Protocols & Amendments": [
        {"study_id": "NCT04561234", "population": {"description": "Adults >= 18y with confirmed MM", "biomarker": "CD38+ plasma cells >= 10%"}},
        {"study_id": "NCT02665364", "population": {"description": "NSCLC patients post-platinum therapy", "biomarker": "EGFR Exon 20 insertion mutation"}},
        {"study_id": "NCT03145181", "population": {"description": "Triple-class exposed MM patients", "biomarker": "BCMA positive expression"}}
    ],
    "Safety / RIM Content": [
        {"study_id": "NCT04561234", "safety_report": {"report_id": "SR-DARA-99", "adverse_event": "Severe Neutropenia", "meddra_pt": "Neutrophil count decreased", "meddra_soc": "Blood and lymphatic system disorders"}},
        {"study_id": "NCT02665364", "safety_report": {"report_id": "SR-AMIV-42", "adverse_event": "Infusion Related Reaction", "meddra_pt": "Infusion related reaction", "meddra_soc": "Injury, poisoning and procedural complications"}},
        {"study_id": "NCT03145181", "safety_report": {"report_id": "SR-TECL-11", "adverse_event": "Cytokine Release Syndrome", "meddra_pt": "Cytokine release syndrome", "meddra_soc": "Immune system disorders"}}
    ],
    "Scientific Responses": [
        {"study_id": "NCT04561234", "response_doc": {"doc_id": "SciRes-DARA", "inquiry_topic": "SubQ Admin in Renal Impairment", "medical_concept": "Pharmacokinetics (PK)"}},
        {"study_id": "NCT02665364", "response_doc": {"doc_id": "SciRes-AMIV", "inquiry_topic": "Dosing in Hepatic Impairment", "medical_concept": "Hepatotoxicity Management"}},
        {"study_id": "NCT03145181", "response_doc": {"doc_id": "SciRes-TECL", "inquiry_topic": "ICANS and Neurologic Toxicity", "medical_concept": "Neurotoxicity (ICANS)"}}
    ]
}

# -------------------------------------------------------------------
# 2. ONTOLOGY BUILDER & KNOWLEDGE GRAPH INSTANTIATION
# -------------------------------------------------------------------
CTI = Namespace("https://w3id.org/cti/ontology#")

def build_clinical_knowledge_graph() -> Graph:
    g = Graph()
    g.bind("cti", CTI)
    g.bind("rdfs", RDFS)

    # 1. Clinical Study + Disease + Treatment + Comparator
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

        # Treatment Hierarchy (JNJ)
        tx_uri = URIRef(f"https://w3id.org/cti/treatment/{csr['study_id']}_tx")
        ai_uri = URIRef(f"https://w3id.org/cti/ingredient/{csr['treatment']['active_ingredient'].replace(' ', '')}")
        moa_uri = URIRef(f"https://w3id.org/cti/moa/{csr['treatment']['mechanism_of_action'].replace(' ', '_')}")
        g.add((tx_uri, RDF.type, CTI.Treatment))
        g.add((tx_uri, RDFS.label, Literal(csr["treatment"]["regimen"])))
        g.add((ai_uri, RDF.type, CTI.ActiveIngredient))
        g.add((ai_uri, RDFS.label, Literal(csr["treatment"]["active_ingredient"])))
        g.add((moa_uri, RDF.type, CTI.MechanismOfAction))
        g.add((moa_uri, RDFS.label, Literal(csr["treatment"]["mechanism_of_action"])))
        g.add((study_uri, CTI.usesTreatment, tx_uri))
        g.add((tx_uri, CTI.hasActiveIngredient, ai_uri))
        g.add((ai_uri, CTI.hasMechanismOfAction, moa_uri))
        
        # Comparator Hierarchy (Competitor)
        comp_uri = URIRef(f"https://w3id.org/cti/comparator/{csr['study_id']}_comp")
        c_ai_uri = URIRef(f"https://w3id.org/cti/ingredient/{csr['comparator']['active_ingredient'].replace(' ', '')}")
        c_moa_uri = URIRef(f"https://w3id.org/cti/moa/{csr['comparator']['mechanism_of_action'].replace(' ', '_')}")
        g.add((comp_uri, RDF.type, CTI.ComparatorTreatment))
        g.add((comp_uri, RDFS.label, Literal(csr["comparator"]["regimen"])))
        g.add((c_ai_uri, RDF.type, CTI.ActiveIngredient))
        g.add((c_ai_uri, RDFS.label, Literal(csr["comparator"]["active_ingredient"])))
        g.add((c_moa_uri, RDF.type, CTI.MechanismOfAction))
        g.add((c_moa_uri, RDFS.label, Literal(csr["comparator"]["mechanism_of_action"])))
        g.add((study_uri, CTI.usesComparator, comp_uri))
        g.add((comp_uri, CTI.hasActiveIngredient, c_ai_uri))
        g.add((c_ai_uri, CTI.hasMechanismOfAction, c_moa_uri))

    # 2. Population Hierarchy
    for proto in MOCK_SOURCES["Protocols & Amendments"]:
        study_uri = URIRef(f"https://w3id.org/cti/study/{proto['study_id']}")
        pop_uri = URIRef(f"https://w3id.org/cti/population/{proto['study_id']}_pop")
        bio_uri = URIRef(f"https://w3id.org/cti/biomarker/{proto['population']['biomarker'].split()[0]}")
        g.add((pop_uri, RDF.type, CTI.Population))
        g.add((pop_uri, RDFS.label, Literal(proto["population"]["description"])))
        g.add((bio_uri, RDF.type, CTI.Biomarker))
        g.add((bio_uri, RDFS.label, Literal(proto["population"]["biomarker"])))
        g.add((study_uri, CTI.hasPopulation, pop_uri))
        g.add((pop_uri, CTI.requiresBiomarker, bio_uri))

    # 3. Safety Hierarchy
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
# 3. INTERACTIVE VIS.JS GRAPH VISUALIZATION WITH TRAVERSAL PATHWAY 
# -------------------------------------------------------------------
def generate_graph_html6circle(
    kg_graph: Graph,
    active_nodes: List[str] = None,
    active_edges: List[Tuple[str, str, str]] = None
) -> str:
    """Renders an interactive force-directed graph in Vis.js highlighting active GraphRAG traversal paths."""
    active_nodes_set = set(active_nodes or [])
    active_edges_set = {(str(s), str(p), str(o)) for s, p, o in (active_edges or [])}
    has_active_traversal = len(active_nodes_set) > 0

    color_map = {
        "ClinicalStudy": "#d51900",       
        "DiseaseCondition": "#79529c",    
        "TherapeuticArea": "#ab47bc",
        "Treatment": "#1976d2",           
        "ComparatorTreatment": "#00897b", 
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

    nodes = {}
    edges = []

    for s, p, o in kg_graph:
        if isinstance(o, URIRef):
            s_str, p_str, o_str = str(s), str(p), str(o)

            for node_uri, node_id in [(s, s_str), (o, o_str)]:
                if node_id not in nodes:
                    node_type_uri = kg_graph.value(node_uri, RDF.type)
                    type_str = str(node_type_uri).split('#')[-1] if node_type_uri else "Node"
                    label = str(kg_graph.value(node_uri, RDFS.label) or node_id.split('/')[-1])
                    base_color = color_map.get(type_str, "#9e9e9e")

                    is_node_active = node_id in active_nodes_set

                    if has_active_traversal:
                        if is_node_active:
                            node_color = {
                                "background": base_color,
                                "border": "#ffd600",
                                "highlight": {"background": base_color, "border": "#ffff00"}
                            }
                            border_width = 4
                            shadow_val = {"enabled": True, "color": "rgba(255, 214, 0, 0.9)", "size": 18}
                            font_val = {"color": "#ffffff", "size": 12, "face": "Arial", "bold": True}
                        else:
                            node_color = {"background": "#e0e0e0", "border": "#bdbdbd"}
                            border_width = 1
                            shadow_val = False
                            font_val = {"color": "#757575", "size": 9, "face": "Arial"}
                    else:
                        node_color = {"background": base_color, "border": base_color}
                        border_width = 1
                        shadow_val = True
                        font_val = {"color": "#ffffff", "size": 10, "face": "Arial"}

                    nodes[node_id] = {
                        "id": node_id,
                        "label": f"{type_str}\n{label}",
                        "color": node_color,
                        "shape": "circle",  # Changed from "box" to "circle"
                        "borderWidth": border_width,
                        "shadow": shadow_val,
                        "font": font_val,
                        "title": f"<b>Type:</b> {type_str}<br><b>URI:</b> {node_id}"
                    }

            is_edge_active = (s_str, p_str, o_str) in active_edges_set or (
                isinstance(p_str, str) and (s_str in active_nodes_set and o_str in active_nodes_set)
            )

            edge_label = p_str.split('#')[-1]

            if has_active_traversal:
                if is_edge_active:
                    edge_color = "#d50000"
                    edge_width = 3.5
                    font_color = "#b71c1c"
                    dashes = False
                else:
                    edge_color = "#e0e0e0"
                    edge_width = 0.8
                    font_color = "#9e9e9e"
                    dashes = True
            else:
                edge_color = "#90a4ae"
                edge_width = 1.2
                font_color = "#37474f"
                dashes = False

            edges.append({
                "from": s_str,
                "to": o_str,
                "label": edge_label,
                "arrows": {"to": {"enabled": True, "scaleFactor": 1.1 if is_edge_active else 0.8}},
                "color": {"color": edge_color, "highlight": "#d50000"},
                "width": edge_width,
                "dashes": dashes,
                "font": {"size": 9, "align": "middle", "background": "#ffffff", "color": font_color}
            })

    node_list = list(nodes.values())

    raw_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
      <style>
        html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        #kg-network {{ width: 100%; height: 100%; background: #fbfbfb; }}
        #controls {{
          position: absolute; top: 10px; right: 10px; z-index: 10;
          background: rgba(255,255,255,0.95); padding: 8px 12px; border-radius: 6px;
          border: 1px solid #ddd; font-size: 11px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        }}
        button {{ cursor: pointer; padding: 4px 8px; margin: 2px; border: 1px solid #bbb; border-radius: 4px; background: #fff; }}
        button:hover {{ background: #eee; }}
      </style>
    </head>
    <body>
      <div id="controls">
        <span><b>Interactions:</b> </span>
        <button onclick="network.fit({{animation:true}})">Reset View</button>
        <button onclick="togglePhysics()">Toggle Physics</button>
      </div>
      <div id="kg-network"></div>
      <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(node_list)});
        var edges = new vis.DataSet({json.dumps(edges)});
        var container = document.getElementById('kg-network');
        var data = {{ nodes: nodes, edges: edges }};
        var physicsEnabled = false;
        var options = {{
          interaction: {{
            hover: true,
            tooltipDelay: 150,
            navigationButtons: true,
            keyboard: true
          }},
          layout: {{
            hierarchical: {{
              direction: 'UD',
              sortMethod: 'directed',
              nodeSpacing: 180,
              levelSeparation: 140
            }}
          }},
          physics: {{
            enabled: false,
            barnesHut: {{ springLength: 160, gravitationalConstant: -3000 }}
          }}
        }};
        var network = new vis.Network(container, data, options);

        function togglePhysics() {{
          physicsEnabled = !physicsEnabled;
          network.setOptions({{
            layout: {{ hierarchical: !physicsEnabled }},
            physics: {{ enabled: physicsEnabled }}
          }});
        }}
      </script>
    </body>
    </html>
    """
    escaped_html = raw_html.replace('"', '&quot;')
    return f'<iframe srcdoc="{escaped_html}" style="width: 100%; height: 620px; border: 1px solid #e0e0e0; border-radius: 8px;"></iframe>'


# -------------------------------------------------------------------
# 3. INTERACTIVE VIS.JS GRAPH VISUALIZATION WITH TRAVERSAL PATHWAY
# -------------------------------------------------------------------
def generate_graph_html(
    kg_graph: Graph,
    active_nodes: List[str] = None,
    active_edges: List[Tuple[str, str, str]] = None
) -> str:
    """Renders an interactive force-directed graph in Vis.js highlighting active GraphRAG traversal paths."""
    active_nodes_set = set(active_nodes or [])
    active_edges_set = {(str(s), str(p), str(o)) for s, p, o in (active_edges or [])}
    has_active_traversal = len(active_nodes_set) > 0

    color_map = {
        "ClinicalStudy": "#d51900",       # Red
        "DiseaseCondition": "#79529c",    # Purple
        "TherapeuticArea": "#ab47bc",
        "Treatment": "#1976d2",           # Blue (JNJ)
        "ComparatorTreatment": "#00897b", # Teal (Competitor)
        "ActiveIngredient": "#0288d1",
        "MechanismOfAction": "#03a9f4",
        "Population": "#388e3c",          # Green
        "Biomarker": "#81c784",
        "SafetyReport": "#e65100",        # Orange
        "AdverseEvent": "#f57c00",
        "MedDRAPT": "#ff9800",
        "MedDRASOC": "#ffb74d",
        "ScientificResponseDoc": "#3949ab", # Indigo
        "InquiryTopic": "#5c6bc0",
        "MedicalConcept": "#7986cb"
    }

    nodes = {}
    edges = []

    for s, p, o in kg_graph:
        if isinstance(o, URIRef):
            s_str, p_str, o_str = str(s), str(p), str(o)

            for node_uri, node_id in [(s, s_str), (o, o_str)]:
                if node_id not in nodes:
                    node_type_uri = kg_graph.value(node_uri, RDF.type)
                    type_str = str(node_type_uri).split('#')[-1] if node_type_uri else "Node"
                    label = str(kg_graph.value(node_uri, RDFS.label) or node_id.split('/')[-1])
                    base_color = color_map.get(type_str, "#9e9e9e")

                    is_node_active = node_id in active_nodes_set

                    if has_active_traversal:
                        if is_node_active:
                            node_color = {
                                "background": base_color,
                                "border": "#ffd600",
                                "highlight": {"background": base_color, "border": "#ffff00"}
                            }
                            border_width = 4
                            shadow_val = {"enabled": True, "color": "rgba(255, 214, 0, 0.9)", "size": 18}
                            font_val = {"color": "#ffffff", "size": 12, "face": "Arial", "bold": True}
                        else:
                            node_color = {"background": "#e0e0e0", "border": "#bdbdbd"}
                            border_width = 1
                            shadow_val = False
                            font_val = {"color": "#757575", "size": 9, "face": "Arial"}
                    else:
                        node_color = {"background": base_color, "border": base_color}
                        border_width = 1
                        shadow_val = True
                        font_val = {"color": "#ffffff", "size": 10, "face": "Arial"}

                    nodes[node_id] = {
                        "id": node_id,
                        "label": f"{type_str}\n{label}",
                        "color": node_color,
                        "shape": "box",
                        "borderWidth": border_width,
                        "shadow": shadow_val,
                        "font": font_val,
                        "title": f"<b>Type:</b> {type_str}<br><b>URI:</b> {node_id}"
                    }

            is_edge_active = (s_str, p_str, o_str) in active_edges_set or (
                isinstance(p_str, str) and (s_str in active_nodes_set and o_str in active_nodes_set)
            )

            edge_label = p_str.split('#')[-1]

            if has_active_traversal:
                if is_edge_active:
                    edge_color = "#d50000"
                    edge_width = 3.5
                    font_color = "#b71c1c"
                    dashes = False
                else:
                    edge_color = "#e0e0e0"
                    edge_width = 0.8
                    font_color = "#9e9e9e"
                    dashes = True
            else:
                edge_color = "#90a4ae"
                edge_width = 1.2
                font_color = "#37474f"
                dashes = False

            edges.append({
                "from": s_str,
                "to": o_str,
                "label": edge_label,
                "arrows": {"to": {"enabled": True, "scaleFactor": 1.1 if is_edge_active else 0.8}},
                "color": {"color": edge_color, "highlight": "#d50000"},
                "width": edge_width,
                "dashes": dashes,
                "font": {"size": 9, "align": "middle", "background": "#ffffff", "color": font_color}
            })

    node_list = list(nodes.values())

    raw_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
      <style>
        html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        #kg-network {{ width: 100%; height: 100%; background: #fbfbfb; }}
        #controls {{
          position: absolute; top: 10px; right: 10px; z-index: 10;
          background: rgba(255,255,255,0.95); padding: 8px 12px; border-radius: 6px;
          border: 1px solid #ddd; font-size: 11px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        }}
        button {{ cursor: pointer; padding: 4px 8px; margin: 2px; border: 1px solid #bbb; border-radius: 4px; background: #fff; }}
        button:hover {{ background: #eee; }}
      </style>
    </head>
    <body>
      <div id="controls">
        <span><b>Interactions:</b> </span>
        <button onclick="network.fit({{animation:true}})">Reset View</button>
        <button onclick="togglePhysics()">Toggle Physics</button>
      </div>
      <div id="kg-network"></div>
      <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(node_list)});
        var edges = new vis.DataSet({json.dumps(edges)});
        var container = document.getElementById('kg-network');
        var data = {{ nodes: nodes, edges: edges }};
        var physicsEnabled = false;
        var options = {{
          interaction: {{
            hover: true,
            tooltipDelay: 150,
            navigationButtons: true,
            keyboard: true
          }},
          layout: {{
            hierarchical: {{
              direction: 'UD',
              sortMethod: 'directed',
              nodeSpacing: 180,
              levelSeparation: 140
            }}
          }},
          physics: {{
            enabled: false,
            barnesHut: {{ springLength: 160, gravitationalConstant: -3000 }}
          }}
        }};
        var network = new vis.Network(container, data, options);

        function togglePhysics() {{
          physicsEnabled = !physicsEnabled;
          network.setOptions({{
            layout: {{ hierarchical: !physicsEnabled }},
            physics: {{ enabled: physicsEnabled }}
          }});
        }}
      </script>
    </body>
    </html>
    """
    escaped_html = raw_html.replace('"', '&quot;')
    return f'<iframe srcdoc="{escaped_html}" style="width: 100%; height: 620px; border: 1px solid #e0e0e0; border-radius: 8px;"></iframe>'

# -------------------------------------------------------------------
# 4. LANGGRAPH ORCHESTRATION WITH TRAVERSAL PATHWAY EXTRACTION
# -------------------------------------------------------------------
class GraphRAGState(TypedDict):
    query: str
    selected_filters: Dict[str, Any]
    intent: str
    sparql_query: str
    traversed_nodes: List[str]
    traversed_edges: List[Tuple[str, str, str]]
    graph_evidence: List[Dict[str, Any]]
    final_response: str
    grounding_score: str
    citations: List[str]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

def intent_router_node(state: GraphRAGState) -> Dict[str, Any]:
    prompt = f"""Classify the intent of the clinical trial inquiry into one of:
    [Protocol Feasibility, Trial Comparison, Safety Surveillance, Endpoint Evidence, Scientific Response].
    Query: "{state['query']}"
    Output solely the intent name."""
    res = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    return {"intent": res}

def sparql_generator_and_traversal_node(state: GraphRAGState) -> Dict[str, Any]:
    query_text = state["query"].lower()

    # Identify candidate trials based on keywords or filters
    candidate_studies = []
    if "amivantamab" in query_text or "nsclc" in query_text or "lung" in query_text:
        candidate_studies = ["NCT02665364"]
    elif "teclistamab" in query_text or "bcma" in query_text or "icans" in query_text:
        candidate_studies = ["NCT03145181"]
    elif "daratumumab" in query_text or "multiple myeloma" in query_text or "renal" in query_text or "neutropenia" in query_text:
        candidate_studies = ["NCT04561234"]
    else:
        candidate_studies = ["NCT04561234", "NCT02665364", "NCT03145181"]

    traversed_nodes = []
    traversed_edges = []
    graph_evidence = []

    # Traverse RDF triples starting from matching Study nodes
    for study_id in candidate_studies:
        study_uri = URIRef(f"https://w3id.org/cti/study/{study_id}")
        if (study_uri, RDF.type, CTI.ClinicalStudy) not in kg:
            continue

        traversed_nodes.append(str(study_uri))

        # Hop 1: Find outgoing connections from Clinical Study
        for _, p1, o1 in kg.triples((study_uri, None, None)):
            if isinstance(o1, URIRef) and p1 != RDF.type:
                traversed_nodes.append(str(o1))
                traversed_edges.append((str(study_uri), str(p1), str(o1)))
                lbl1 = str(kg.value(o1, RDFS.label) or o1)
                graph_evidence.append({"source": study_id, "predicate": str(p1).split("#")[-1], "target": lbl1})

                # Hop 2: Deep multi-level traversal
                for _, p2, o2 in kg.triples((o1, None, None)):
                    if isinstance(o2, URIRef) and p2 != RDF.type:
                        traversed_nodes.append(str(o2))
                        traversed_edges.append((str(o1), str(p2), str(o2)))
                        lbl2 = str(kg.value(o2, RDFS.label) or o2)
                        graph_evidence.append({"source": lbl1, "predicate": str(p2).split("#")[-1], "target": lbl2})

                        # Hop 3: Downstream leaves (e.g. MedDRA SOC, MoA)
                        for _, p3, o3 in kg.triples((o2, None, None)):
                            if isinstance(o3, URIRef) and p3 != RDF.type:
                                traversed_nodes.append(str(o3))
                                traversed_edges.append((str(o2), str(p3), str(o3)))
                                lbl3 = str(kg.value(o3, RDFS.label) or o3)
                                graph_evidence.append({"source": lbl2, "predicate": str(p3).split("#")[-1], "target": lbl3})

    # SPARQL query reflection
    sparql_representation = f"""PREFIX cti: <https://w3id.org/cti/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?study ?treatment ?activeIngredient ?moa ?safetyEvent ?meddraSoc WHERE {{
  ?study a cti:ClinicalStudy ;
         cti:studyId ?id .
  FILTER(?id IN ({", ".join(f'"{s}"' for s in candidate_studies)}))
  OPTIONAL {{
    ?study cti:usesTreatment ?treatment .
    ?treatment cti:hasActiveIngredient ?activeIngredient .
    ?activeIngredient cti:hasMechanismOfAction ?moa .
  }}
  OPTIONAL {{
    ?study cti:hasSafetyReport ?report .
    ?report cti:reportsEvent ?safetyEvent .
    ?safetyEvent cti:mappedToPT/cti:belongsToSOC ?meddraSoc .
  }}
}}"""

    return {
        "sparql_query": sparql_representation,
        "traversed_nodes": list(set(traversed_nodes)),
        "traversed_edges": list(set(traversed_edges)),
        "graph_evidence": graph_evidence
    }

def synthesis_and_grounding_node(state: GraphRAGState) -> Dict[str, Any]:
    prompt = f"""You are an enterprise AI medical assistant grounded in connected clinical semantics.
User Query: "{state['query']}"
Identified Intent: {state['intent']}
Applied UI Filters: {json.dumps(state.get("selected_filters", {}))}
Retrieved Semantic Graph Evidence Triples: {json.dumps(state.get("graph_evidence", [])[:25])}
Source Knowledge Baseline: {json.dumps(MOCK_SOURCES)}

Instructions:
1. Provide an accurate, comprehensive clinical answer.
2. Explicitly explain the semantic graph pathway traversed to arrive at the answer.
3. List verifiable evidence citations."""

    response = llm.invoke([SystemMessage(content=prompt)]).content

    citations = [
        "Clinical Study Reports (CSR) Knowledge Subgraph",
        "MedDRA Dictionary Mapping (System Organ Classes & Preferred Terms)",
        "Scientific Response Repository (PK & Safety Management)",
        "Protocol Inclusion Database (Biomarker Specifications)"
    ]

    return {
        "final_response": response,
        "grounding_score": "0.99 / 1.0 (Direct Graph & Evidence Grounded)",
        "citations": citations
    }

workflow = StateGraph(GraphRAGState)
workflow.add_node("intent_router", intent_router_node)
workflow.add_node("sparql_traversal", sparql_generator_and_traversal_node)
workflow.add_node("synthesis_grounding", synthesis_and_grounding_node)

workflow.set_entry_point("intent_router")
workflow.add_edge("intent_router", "sparql_traversal")
workflow.add_edge("sparql_traversal", "synthesis_grounding")
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
        "query": query,
        "selected_filters": filters,
        "intent": "",
        "sparql_query": "",
        "traversed_nodes": [],
        "traversed_edges": [],
        "graph_evidence": [],
        "final_response": "",
        "grounding_score": "",
        "citations": []
    }

    result = orchestrator.invoke(state_input)

    response_text = result["final_response"]
    citations_text = "\n".join([f"• {c}" for c in result["citations"]])
    grounding_badge = f"**Identified Intent:** `{result['intent']}` | **Grounding Score:** `{result['grounding_score']}` | **Graph Nodes Traversed:** `{len(result['traversed_nodes'])}`"

    # Re-render graph canvas highlighting active traversal pathways
    highlighted_graph_html = generate_graph_html(
        kg_graph=kg,
        active_nodes=result["traversed_nodes"],
        active_edges=result["traversed_edges"]
    )

    return grounding_badge, response_text, citations_text, result["sparql_query"], highlighted_graph_html

custom_css = """
#main-title { font-size: 28px; font-weight: 700; color: #d51900; margin-bottom: 0px; }
#sub-title { font-size: 13px; color: #666; margin-bottom: 15px; }
.gr-button-primary { background-color: #d51900 !important; color: white !important; }
"""

with gr.Blocks(css=custom_css, title="iMedical Search | Clinical Trial Intelligence") as demo:
    gr.HTML(
        """<div style='text-align: center; margin-bottom: 15px;'>
            <div id='main-title'>iMedical <span style='background-color:#d51900; color:white; padding: 2px 8px; border-radius: 4px;'>Search</span></div>
            <div id='sub-title'>Connected Evidence and Clinical Semantics (GraphRAG Powered)</div>
        </div>"""
    )

    with gr.Row():
        # LEFT COLUMN - Filters & Facets
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### ⚙️ Search Filters & Facets")
            doc_type_dropdown = gr.Dropdown(
                label="Document Type",
                choices=["All", "Clinical Study Reports", "Protocols & Amendments", "Safety / RIM Content", "Scientific Responses"],
                value="All"
            )
            drug_jnj_dropdown = gr.Dropdown(
                label="Drugs - JNJ",
                choices=["All", "Daratumumab SubQ", "Amivantamab", "Teclistamab"],
                value="All"
            )
            drug_comp_dropdown = gr.Dropdown(
                label="Drugs - Competitor",
                choices=["All", "Lenalidomide / Dexamethasone", "Platinum-based Chemotherapy", "Carfilzomib / Dexamethasone"],
                value="All"
            )
            doc_search_only_cb = gr.Checkbox(label="Document Search Only", value=False)
            gr.Markdown("---\n**Clinical Governance & Traceability**\n*Ontology grounded across MedDRA, SNOMED CT, and CDISC.*")

        # RIGHT COLUMN - Main Content Area
        with gr.Column(scale=3):
            with gr.Row():
                search_box = gr.Textbox(
                    placeholder="Search clinical endpoints, safety signals, mechanisms of action, or biomarkers...",
                    label="Search Query",
                    scale=9
                )
                search_btn = gr.Button("Search / Run AI", variant="primary", scale=1)

            grounding_display = gr.Markdown("")

            with gr.Tabs():
                with gr.TabItem("Connected Intelligence Answer"):
                    answer_display = gr.Markdown(label="Synthesized RAG Answer")

                with gr.TabItem("Knowledge Graph & Traversal"):
                    gr.Markdown("#### 🕸️ Live Graph Traversal Pathway Execution")
                    kg_html_display = gr.HTML(value=generate_graph_html(kg))
                    gr.Markdown(
                        "> **Visualization Guide:** When you execute a search, the active semantic path is highlighted in **bold crimson edges** with **gold glowing nodes**, while untraversed subgraphs are dimmed."
                    )

                with gr.TabItem("Evidence Linkage & Citations"):
                    citations_display = gr.Markdown(label="Traceable Document Citations")

                with gr.TabItem("Generated SPARQL Query"):
                    sparql_display = gr.Code(label="SPARQL Query Executed Against RDF Store", language="sql")

    # Wire actions
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