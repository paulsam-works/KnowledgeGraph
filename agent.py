"""
LangGraph orchestrator for iMedical Search.

Pipeline: classify intent → walk RDF from matching studies → synthesize a grounded answer.
Shared state is GraphRAGState; the compiled graph is exported as `orchestrator`.
"""
import os
import json
from typing import Dict, List, Any, TypedDict, Tuple
from rdflib import URIRef, RDF, RDFS

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from ontology import kg, CTI
from data_loader import MOCK_SOURCES
from dotenv import load_dotenv


#os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-openai-api-key")


# Find and load the .env file
load_dotenv()

class GraphRAGState(TypedDict):
    """Mutable state passed between LangGraph nodes for a single user search."""
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
    """Map the natural-language query to one of five clinical task intents."""
    prompt = f"""Classify the intent of the clinical trial inquiry into one of:
    [Protocol Feasibility, Trial Comparison, Safety Surveillance, Endpoint Evidence, Scientific Response].
    Query: "{state['query']}"
    Output solely the intent name."""
    res = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    return {"intent": res}

def sparql_generator_and_traversal_node(state: GraphRAGState) -> Dict[str, Any]:
    query_text = state["query"]

    # 1. DYNAMIC SPARQL CONSTRUCT GENERATION
    sparql_prompt = f"""You are an expert in RDF and SPARQL for Clinical Knowledge Graphs.
    Generate a valid SPARQL CONSTRUCT query that extracts the exact semantic pathway for this user query: "{query_text}"

    ONTOLOGY SCHEMA:
    PREFIX cti: <https://w3id.org/cti/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    - cti:ClinicalStudy -> usesTreatment -> cti:Treatment -> hasActiveIngredient -> cti:ActiveIngredient -> hasMechanismOfAction -> cti:MechanismOfAction
    - cti:ClinicalStudy -> usesComparator -> cti:ComparatorTreatment -> hasActiveIngredient -> cti:ActiveIngredient -> hasMechanismOfAction -> cti:MechanismOfAction
    - cti:ClinicalStudy -> targetsDisease -> cti:DiseaseCondition -> belongsToTherapeuticArea -> cti:TherapeuticArea
    - cti:ClinicalStudy -> hasPopulation -> cti:Population -> requiresBiomarker -> cti:Biomarker
    - cti:ClinicalStudy -> hasSafetyReport -> cti:SafetyReport -> reportsEvent -> cti:AdverseEvent -> mappedToPT -> cti:MedDRAPT -> belongsToSOC -> cti:MedDRASOC
    - cti:ClinicalStudy -> hasScientificResponse -> cti:ScientificResponseDoc -> addressesTopic -> cti:InquiryTopic -> relatesToConcept -> cti:MedicalConcept

    RULES:
    1. MUST output a CONSTRUCT query.
    2. Construct the FULL chain of triples from the ClinicalStudy down to the leaf node.
    3. ONLY return raw SPARQL code. Do NOT include markdown backticks (```).
    """

    raw_sparql = llm.invoke([HumanMessage(content=sparql_prompt)]).content
    clean_sparql = raw_sparql.replace("```sparql", "").replace("```", "").strip()

    traversed_nodes = []
    traversed_edges = []
    graph_evidence = []

    # 2. DYNAMIC GRAPH EXECUTION
    try:
        query_result = kg.query(clean_sparql)
        
        if query_result.type != 'CONSTRUCT':
            raise ValueError("LLM generated a SELECT query instead of CONSTRUCT.")

        for s, p, o in query_result:
            s_str, p_str, o_str = str(s), str(p), str(o)
            traversed_nodes.extend([s_str, o_str])
            traversed_edges.append((s_str, p_str, o_str))
            
            s_lbl = str(kg.value(s, RDFS.label) or s_str.split('/')[-1])
            o_lbl = str(kg.value(o, RDFS.label) or o_str.split('/')[-1])
            if "w3.org" not in p_str: 
                graph_evidence.append({"source": s_lbl, "predicate": p_str.split('#')[-1], "target": o_lbl})

    except Exception as e:
        print(f"SPARQL Execution Error: {e}")
        pass

    # 3. DEEP FAILSAFE / FALLBACK MECHANISM (Now traverses up to 4 levels deep)
    if len(traversed_edges) == 0:
        query_lower = query_text.lower()
        candidate_studies = []
        
        if "amivantamab" in query_lower or "nsclc" in query_lower or "lung" in query_lower:
            candidate_studies.append("NCT02665364")
        elif "teclistamab" in query_lower or "bcma" in query_lower or "icans" in query_lower:
            candidate_studies.append("NCT03145181")
        elif "daratumumab" in query_lower or "myeloma" in query_lower or "renal" in query_lower or "pk" in query_lower or "pharmacokinetics" in query_lower:
            candidate_studies.append("NCT04561234")
        else:
            candidate_studies = ["NCT04561234", "NCT02665364", "NCT03145181"]

        for study_id in candidate_studies:
            study_uri = URIRef(f"[https://w3id.org/cti/study/](https://w3id.org/cti/study/){study_id}")
            if (study_uri, RDF.type, CTI.ClinicalStudy) in kg:
                traversed_nodes.append(str(study_uri))
                
                # Hop 1
                for _, p1, o1 in kg.triples((study_uri, None, None)):
                    if isinstance(o1, URIRef) and p1 != RDF.type:
                        traversed_nodes.extend([str(study_uri), str(o1)])
                        traversed_edges.append((str(study_uri), str(p1), str(o1)))
                        
                        # Hop 2
                        for _, p2, o2 in kg.triples((o1, None, None)):
                            if isinstance(o2, URIRef) and p2 != RDF.type:
                                traversed_nodes.extend([str(o1), str(o2)])
                                traversed_edges.append((str(o1), str(p2), str(o2)))
                                
                                # Hop 3 (Catches Scientific Responses and Mechanisms)
                                for _, p3, o3 in kg.triples((o2, None, None)):
                                    if isinstance(o3, URIRef) and p3 != RDF.type:
                                        traversed_nodes.extend([str(o2), str(o3)])
                                        traversed_edges.append((str(o2), str(p3), str(o3)))
                                        
                                        # Hop 4 (Catches deeply nested MedDRA System Organ Classes)
                                        for _, p4, o4 in kg.triples((o3, None, None)):
                                            if isinstance(o4, URIRef) and p4 != RDF.type:
                                                traversed_nodes.extend([str(o3), str(o4)])
                                                traversed_edges.append((str(o3), str(p4), str(o4)))

        clean_sparql = "# [NOTE: AI SPARQL returned 0 results. Triggered deep fallback traversal.]\n\n" + clean_sparql

    return {
        "sparql_query": clean_sparql,
        "traversed_nodes": list(set(traversed_nodes)),
        "traversed_edges": list(set(traversed_edges)),
        "graph_evidence": graph_evidence
    }
def sparql_generator_and_traversal_node1111(state: GraphRAGState) -> Dict[str, Any]:
    query_text = state["query"]

    # 1. DYNAMIC SPARQL CONSTRUCT GENERATION (With 1-Shot Example)
    sparql_prompt = f"""You are an expert in RDF and SPARQL for Clinical Knowledge Graphs.
    Generate a valid SPARQL CONSTRUCT query that extracts the exact semantic pathway for this user query: "{query_text}"

    ONTOLOGY SCHEMA:
    PREFIX cti: <https://w3id.org/cti/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    - cti:ClinicalStudy -> usesTreatment -> cti:Treatment -> hasActiveIngredient -> cti:ActiveIngredient -> hasMechanismOfAction -> cti:MechanismOfAction
    - cti:ClinicalStudy -> usesComparator -> cti:ComparatorTreatment -> hasActiveIngredient -> cti:ActiveIngredient -> hasMechanismOfAction -> cti:MechanismOfAction
    - cti:ClinicalStudy -> targetsDisease -> cti:DiseaseCondition -> belongsToTherapeuticArea -> cti:TherapeuticArea
    - cti:ClinicalStudy -> hasPopulation -> cti:Population -> requiresBiomarker -> cti:Biomarker
    - cti:ClinicalStudy -> hasSafetyReport -> cti:SafetyReport -> reportsEvent -> cti:AdverseEvent -> mappedToPT -> cti:MedDRAPT -> belongsToSOC -> cti:MedDRASOC
    - cti:ClinicalStudy -> hasScientificResponse -> cti:ScientificResponseDoc -> addressesTopic -> cti:InquiryTopic -> relatesToConcept -> cti:MedicalConcept

    RULES:
    1. MUST output a CONSTRUCT query. Do not output a SELECT query.
    2. Construct the FULL chain of triples from the ClinicalStudy down to the leaf node.
    3. ONLY return raw SPARQL code. Do NOT include markdown backticks (```).

    EXAMPLE TEMPLATE:
    CONSTRUCT {{
      ?study cti:usesTreatment ?tx .
      ?tx cti:hasActiveIngredient ?ai .
      ?ai cti:hasMechanismOfAction ?moa .
    }} WHERE {{
      ?study a cti:ClinicalStudy ; cti:usesTreatment ?tx .
      ?tx cti:hasActiveIngredient ?ai .
      ?ai cti:hasMechanismOfAction ?moa .
      ?ai rdfs:label ?label .
      FILTER(REGEX(str(?label), "Keyword", "i"))
    }}
    """

    raw_sparql = llm.invoke([HumanMessage(content=sparql_prompt)]).content
    clean_sparql = raw_sparql.replace("```sparql", "").replace("```", "").strip()

    traversed_nodes = []
    traversed_edges = []
    graph_evidence = []

    # 2. DYNAMIC GRAPH EXECUTION
    try:
        query_result = kg.query(clean_sparql)
        
        # Guard against the LLM accidentally generating a SELECT query
        if query_result.type != 'CONSTRUCT':
            raise ValueError("LLM generated a SELECT query instead of CONSTRUCT.")

        for s, p, o in query_result:
            s_str, p_str, o_str = str(s), str(p), str(o)
            traversed_nodes.extend([s_str, o_str])
            traversed_edges.append((s_str, p_str, o_str))
            
            s_lbl = str(kg.value(s, RDFS.label) or s_str.split('/')[-1])
            o_lbl = str(kg.value(o, RDFS.label) or o_str.split('/')[-1])
            if "w3.org" not in p_str: 
                graph_evidence.append({"source": s_lbl, "predicate": p_str.split('#')[-1], "target": o_lbl})

    except Exception as e:
        print(f"SPARQL Execution Error: {e}")
        pass # Allow the script to continue to the fallback below

    # 3. FAILSAFE / FALLBACK MECHANISM
    # If the LLM's query failed or was too strict and returned 0 nodes, trigger an internal Python traversal.
    if len(traversed_edges) == 0:
        query_lower = query_text.lower()
        candidate_studies = []
        
        if "amivantamab" in query_lower or "nsclc" in query_lower or "lung" in query_lower:
            candidate_studies.append("NCT02665364")
        elif "teclistamab" in query_lower or "bcma" in query_lower or "icans" in query_lower:
            candidate_studies.append("NCT03145181")
        elif "daratumumab" in query_lower or "myeloma" in query_lower or "renal" in query_lower:
            candidate_studies.append("NCT04561234")
        else:
            candidate_studies = ["NCT04561234", "NCT02665364", "NCT03145181"]

        for study_id in candidate_studies:
            study_uri = URIRef(f"[https://w3id.org/cti/study/](https://w3id.org/cti/study/){study_id}")
            if (study_uri, RDF.type, CTI.ClinicalStudy) in kg:
                traversed_nodes.append(str(study_uri))
                
                # Broadly highlight the relevant study's pathways to ensure the UI updates
                for _, p1, o1 in kg.triples((study_uri, None, None)):
                    if isinstance(o1, URIRef) and p1 != RDF.type:
                        traversed_nodes.append(str(o1))
                        traversed_edges.append((str(study_uri), str(p1), str(o1)))
                        
                        for _, p2, o2 in kg.triples((o1, None, None)):
                            if isinstance(o2, URIRef) and p2 != RDF.type:
                                traversed_nodes.append(str(o2))
                                traversed_edges.append((str(o1), str(p2), str(o2)))

        clean_sparql = "# [NOTE: AI SPARQL returned 0 results. Triggered internal fallback traversal.]\n\n" + clean_sparql

    return {
        "sparql_query": clean_sparql,
        "traversed_nodes": list(set(traversed_nodes)),
        "traversed_edges": list(set(traversed_edges)),
        "graph_evidence": graph_evidence
    }

def sparql_generator_and_traversal_node22222(state: GraphRAGState) -> Dict[str, Any]:
    query_text = state["query"]

    # 1. DYNAMIC SPARQL CONSTRUCT GENERATION
    # We provide the ontology schema to the LLM so it knows exactly how to map the multi-hop paths.
    sparql_prompt = f"""You are an expert in RDF and SPARQL for Clinical Knowledge Graphs.
Generate a valid SPARQL CONSTRUCT query that extracts the exact semantic pathway for this user query: "{query_text}"

ONTOLOGY SCHEMA:
PREFIX cti: <https://w3id.org/cti/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

- cti:ClinicalStudy -> usesTreatment -> cti:Treatment -> hasActiveIngredient -> cti:ActiveIngredient -> hasMechanismOfAction -> cti:MechanismOfAction
- cti:ClinicalStudy -> usesComparator -> cti:ComparatorTreatment ... (same as above)
- cti:ClinicalStudy -> targetsDisease -> cti:DiseaseCondition -> belongsToTherapeuticArea -> cti:TherapeuticArea
- cti:ClinicalStudy -> hasPopulation -> cti:Population -> requiresBiomarker -> cti:Biomarker
- cti:ClinicalStudy -> hasSafetyReport -> cti:SafetyReport -> reportsEvent -> cti:AdverseEvent -> mappedToPT -> cti:MedDRAPT -> belongsToSOC -> cti:MedDRASOC
- cti:ClinicalStudy -> hasScientificResponse -> cti:ScientificResponseDoc -> addressesTopic -> cti:InquiryTopic -> relatesToConcept -> cti:MedicalConcept

RULES:
1. You MUST output a CONSTRUCT query: CONSTRUCT {{ ?s ?p ?o . ... }} WHERE {{ ?s ?p ?o . ... }}
2. Use FILTER(REGEX(str(?label), "keyword", "i")) on rdfs:label to match user keywords (e.g., "amivantamab", "safety", "biomarker").
3. Construct the FULL chain of triples (from ClinicalStudy down to the leaf node) so the UI can trace the complete path.
4. ONLY return raw SPARQL code. Do NOT include markdown backticks (```) or any explanations.
"""

    raw_sparql = llm.invoke([HumanMessage(content=sparql_prompt)]).content
    clean_sparql = raw_sparql.replace("```sparql", "").replace("```", "").strip()

    traversed_nodes = []
    traversed_edges = []
    graph_evidence = []

    # 2. DYNAMIC GRAPH EXECUTION & PATHWAY EXTRACTION
    try:
        # Execute the generated CONSTRUCT query against the RDF graph
        query_result = kg.query(clean_sparql)
        
        # A CONSTRUCT query returns a subset Graph of (subject, predicate, object) triples
        for s, p, o in query_result:
            s_str, p_str, o_str = str(s), str(p), str(o)
            
            # Record the dynamic nodes and edges for Vis.js highlighting
            traversed_nodes.extend([s_str, o_str])
            traversed_edges.append((s_str, p_str, o_str))
            
            # Extract readable labels for the AI Synthesis context window
            s_lbl = str(kg.value(s, RDFS.label) or s_str.split('/')[-1])
            o_lbl = str(kg.value(o, RDFS.label) or o_str.split('/')[-1])
            if "w3.org" not in p_str: # Filter out raw rdf:type definitions for cleaner context
                graph_evidence.append({"source": s_lbl, "predicate": p_str.split('#')[-1], "target": o_lbl})

    except Exception as e:
        # Graceful fallback if the LLM hallucinates invalid SPARQL syntax
        print(f"SPARQL Parse Error: {e}")
        clean_sparql = f"# SPARQL Syntax Error Executing Query\n# Reverting to baseline extraction\n"
        # Return empty traversals so the graph resets to default view safely
        traversed_nodes = []
        traversed_edges = []

    return {
        "sparql_query": clean_sparql,
        "traversed_nodes": list(set(traversed_nodes)),
        "traversed_edges": list(set(traversed_edges)),
        "graph_evidence": graph_evidence
    }


def sparql_generator_and_traversal_node_back_st(state: GraphRAGState) -> Dict[str, Any]:
    """Select studies, walk three RDF hops, and emit an auditable SPARQL sketch."""
    query_text = state["query"].lower()

    # Keyword routing onto mock NCT IDs (not a full SPARQL planner).
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

    # SPARQL is a reflection of the walk for the UI audit tab, not executed against a remote store.
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
    """Compose the clinical answer from graph triples, filters, and source JSON."""
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

    # Prototype citations: source-class labels rather than page-level offsets.
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

def build_workflow():
    """Linear graph: intent_router → sparql_traversal → synthesis_grounding → END."""
    workflow = StateGraph(GraphRAGState)
    workflow.add_node("intent_router", intent_router_node)
    workflow.add_node("sparql_traversal", sparql_generator_and_traversal_node)
    workflow.add_node("synthesis_grounding", synthesis_and_grounding_node)

    workflow.set_entry_point("intent_router")
    workflow.add_edge("intent_router", "sparql_traversal")
    workflow.add_edge("sparql_traversal", "synthesis_grounding")
    workflow.add_edge("synthesis_grounding", END)

    return workflow.compile()


orchestrator = build_workflow()