"""
iMedical Search — Gradio portal for GraphRAG clinical-trial search.

Wires filters + query into the LangGraph orchestrator, then renders the
synthesized answer, citations, SPARQL, and a highlighted knowledge-graph canvas.
"""
import gradio as gr
from agent import orchestrator, GraphRAGState
from ontology import kg
from visualization import generate_graph_html


def search_pipeline(query: str, doc_type: str, drug_impts: str, drug_comp: str, doc_search_only: bool):
    """Run one GraphRAG search and map results onto Gradio outputs.

    Returns: grounding badge, answer markdown, citations, SPARQL text, graph HTML.
    """
    if not query.strip():
        # Keep the unhighlighted ontology visible so the graph tab is never empty.
        return "", "", "", "Please enter a clinical query.", generate_graph_html(kg)

    filters = {
        "Document Type": doc_type,
        "Drug - IPMTS": drug_impts,
        "Drug - Competitor": drug_comp,
        "Document Search Only": doc_search_only
    }

    # Empty fields are required so LangGraph state keys are always present.
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

    result = orchestrator.invoke(state_input)  # intent → RDF traversal → LLM synthesis

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

# Branding aligned with the iMedical Search mock (crimson primary).
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
            drug_impts_dropdown = gr.Dropdown(
                label="Drugs - IPMTS",
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

            grounding_display = gr.Markdown("")  # intent + grounding score + node count

            with gr.Tabs():
                with gr.TabItem("Connected Intelligence Answer"):
                    answer_display = gr.Markdown(label="Synthesized RAG Answer")

                with gr.TabItem("Knowledge Graph & Traversal"):
                    gr.Markdown("#### 🕸️ Live Graph Traversal Pathway Execution")
                    kg_html_display = gr.HTML(value=generate_graph_html(kg))  # full graph until a search highlights a path
                    gr.Markdown(
                        "> **Visualization Guide:** When you execute a search, the active semantic path is highlighted in **bold crimson edges** with **gold glowing nodes**, while untraversed subgraphs are dimmed."
                    )

                with gr.TabItem("Evidence Linkage & Citations"):
                    citations_display = gr.Markdown(label="Traceable Document Citations")

                with gr.TabItem("Generated SPARQL Query"):
                    sparql_display = gr.Code(label="SPARQL Query Executed Against RDF Store", language="sql")

    # Button click and Enter in the search box both run the same pipeline.
    search_btn.click(
        fn=search_pipeline,
        inputs=[search_box, doc_type_dropdown, drug_impts_dropdown, drug_comp_dropdown, doc_search_only_cb],
        outputs=[grounding_display, answer_display, citations_display, sparql_display, kg_html_display]
    )
    search_box.submit(
        fn=search_pipeline,
        inputs=[search_box, doc_type_dropdown, drug_impts_dropdown, drug_comp_dropdown, doc_search_only_cb],
        outputs=[grounding_display, answer_display, citations_display, sparql_display, kg_html_display]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)  # http://localhost:7860