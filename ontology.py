"""
RDF clinical-trial ontology built from MOCK_SOURCES.

Custom `cti:` namespace (https://w3id.org/cti/ontology#). Studies are hubs
linked to disease, IMPTS/comparator treatments, population, safety, and
scientific-response subgraphs. Module export: `kg`, `CTI`.
"""
from rdflib import Graph, Literal, RDF, Namespace, URIRef
from rdflib.namespace import RDFS
from data_loader import MOCK_SOURCES

CTI = Namespace("https://w3id.org/cti/ontology#")


def build_clinical_knowledge_graph() -> Graph:
    """Materialize the in-memory rdflib Graph used by traversal and visualization."""
    g = Graph()
    g.bind("cti", CTI)
    g.bind("rdfs", RDFS)

    # 1. Clinical Study + Disease + Treatment + Comparator
    for csr in MOCK_SOURCES["Clinical Study Reports"]:
        study_uri = URIRef(f"https://w3id.org/cti/study/{csr['study_id']}")
        g.add((study_uri, RDF.type, CTI.ClinicalStudy))
        g.add((study_uri, RDFS.label, Literal(csr["study_id"])))

        # Disease → therapeutic area (SNOMED code in the disease URI)
        dis_uri = URIRef(f"https://w3id.org/cti/disease/{csr['disease']['snomed_code']}")
        ta_uri = URIRef(f"https://w3id.org/cti/ta/{csr['disease']['therapeutic_area'].replace(' ', '')}")
        g.add((dis_uri, RDF.type, CTI.DiseaseCondition))
        g.add((dis_uri, RDFS.label, Literal(csr["disease"]["term"])))
        g.add((ta_uri, RDF.type, CTI.TherapeuticArea))
        g.add((ta_uri, RDFS.label, Literal(csr["disease"]["therapeutic_area"])))
        g.add((study_uri, CTI.targetsDisease, dis_uri))
        g.add((dis_uri, CTI.belongsToTherapeuticArea, ta_uri))

        # IMPTS regimen → active ingredient → mechanism of action
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
        
        # Comparator / competitor arm (same ingredient → MoA pattern)
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

    # 2. Protocol population + required biomarker (linked back to the same study URI)
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

    # 3. Safety report → AE → MedDRA PT → SOC
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

    # 4. Scientific response doc → inquiry topic → medical concept
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

# Built once at import time; traversal and Vis.js both read this graph.
kg = build_clinical_knowledge_graph()