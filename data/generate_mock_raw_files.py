import os
import json

# Define the directory structure for our raw document data lake
BASE_DIR = "clinical_raw_data"
DIRECTORIES = {
    "csr": f"{BASE_DIR}/Clinical_Study_Reports",
    "protocol": f"{BASE_DIR}/Protocols_and_Amendments",
    "safety": f"{BASE_DIR}/Safety_RIM",
    "scires": f"{BASE_DIR}/Scientific_Responses"
}

# Create the directories
for path in DIRECTORIES.values():
    os.makedirs(path, exist_ok=True)

# ---------------------------------------------------------
# 1. Generate Raw Clinical Study Reports (JSON format)
# ---------------------------------------------------------
csrs = [
    {
        "file_name": "CSR_NCT04561234_Final.json",
        "content": {
            "metadata": {"study_id": "NCT04561234", "status": "Completed"},
            "title_page": "Phase 3 Study of Daratumumab vs Standard of Care in MM",
            "disease_info": {"condition": "Multiple Myeloma", "snomed": "109989006", "ta": "Hematological Malignancy"},
            "interventions": {
                "investigational": {"name": "Daratumumab 1800mg SubQ", "active_moiety": "Daratumumab", "moa": "CD38-directed cytolytic antibody"},
                "control": {"name": "Lenalidomide / Dexamethasone", "active_moiety": "Lenalidomide", "moa": "Immunomodulatory imide drug"}
            }
        }
    },
    {
        "file_name": "CSR_NCT02665364_Interim.json",
        "content": {
            "metadata": {"study_id": "NCT02665364", "status": "Active"},
            "title_page": "Phase 1/2 Study of Amivantamab in Advanced NSCLC",
            "disease_info": {"condition": "Non-Small Cell Lung Cancer (NSCLC)", "snomed": "254637007", "ta": "Solid Tumors"},
            "interventions": {
                "investigational": {"name": "Amivantamab-vmjw 1050mg IV", "active_moiety": "Amivantamab", "moa": "EGFR/MET bispecific antibody"},
                "control": {"name": "Platinum-based Chemotherapy", "active_moiety": "Cisplatin", "moa": "Alkylating antineoplastic agent"}
            }
        }
    }
]

for csr in csrs:
    with open(os.path.join(DIRECTORIES["csr"], csr["file_name"]), "w") as f:
        json.dump(csr["content"], f, indent=4)


# ---------------------------------------------------------
# 2. Generate Raw Protocol Documents (Unstructured TXT format)
# ---------------------------------------------------------
protocols = [
    {
        "file_name": "Protocol_NCT04561234_v3.2.txt",
        "content": """CLINICAL TRIAL PROTOCOL
Study ID: NCT04561234
Amendment: 3.2

4.1 Inclusion Criteria:
- Patients must be Adults >= 18y with confirmed MM.
- Must exhibit CD38+ plasma cells >= 10% on bone marrow aspirate.
"""
    },
    {
        "file_name": "Protocol_NCT02665364_v1.1.txt",
        "content": """CLINICAL TRIAL PROTOCOL
Study ID: NCT02665364

4.1 Inclusion Criteria:
- NSCLC patients post-platinum therapy.
- Molecular testing must confirm EGFR Exon 20 insertion mutation.
"""
    }
]

for protocol in protocols:
    with open(os.path.join(DIRECTORIES["protocol"], protocol["file_name"]), "w") as f:
        f.write(protocol["content"])


# ---------------------------------------------------------
# 3. Generate Raw Safety / RIM Reports (JSON format exported from Argus/RIM)
# ---------------------------------------------------------
safety_reports = [
    {
        "file_name": "Safety_E2B_SR-DARA-99.json",
        "content": {
            "report_id": "SR-DARA-99",
            "study_reference": "NCT04561234",
            "event_details": {
                "reported_term": "Severe Neutropenia",
                "meddra_coding": {
                    "preferred_term": "Neutrophil count decreased",
                    "system_organ_class": "Blood and lymphatic system disorders"
                }
            }
        }
    },
    {
        "file_name": "Safety_E2B_SR-AMIV-42.json",
        "content": {
            "report_id": "SR-AMIV-42",
            "study_reference": "NCT02665364",
            "event_details": {
                "reported_term": "Infusion Related Reaction",
                "meddra_coding": {
                    "preferred_term": "Infusion related reaction",
                    "system_organ_class": "Injury, poisoning and procedural complications"
                }
            }
        }
    }
]

for sr in safety_reports:
    with open(os.path.join(DIRECTORIES["safety"], sr["file_name"]), "w") as f:
        json.dump(sr["content"], f, indent=4)


# ---------------------------------------------------------
# 4. Generate Scientific Responses (Unstructured TXT format)
# ---------------------------------------------------------
sci_res_docs = [
    {
        "file_name": "SRD_SciRes-DARA_Renal.txt",
        "content": """STANDARD RESPONSE DOCUMENT (SRD)
Document ID: SciRes-DARA
Linked Study: NCT04561234

Inquiry Topic: SubQ Admin in Renal Impairment
Medical Concept Tag: Pharmacokinetics (PK)

Response Summary:
Population PK analyses indicate that mild to moderate renal impairment does not significantly alter the exposure of subcutaneous Daratumumab. No dose adjustments are recommended.
"""
    },
    {
        "file_name": "SRD_SciRes-AMIV_Hepatic.txt",
        "content": """STANDARD RESPONSE DOCUMENT (SRD)
Document ID: SciRes-AMIV
Linked Study: NCT02665364

Inquiry Topic: Dosing in Hepatic Impairment
Medical Concept Tag: Hepatotoxicity Management

Response Summary:
Clearance of Amivantamab is not heavily reliant on hepatic pathways, however, intensive monitoring is required for patients exhibiting baseline elevated AST/ALT.
"""
    }
]

for doc in sci_res_docs:
    with open(os.path.join(DIRECTORIES["scires"], doc["file_name"]), "w") as f:
        f.write(doc["content"])

print(f"Successfully generated raw mock files in the '{BASE_DIR}' directory.")