"""
Vis.js knowledge-graph canvas for the Gradio Knowledge Graph tab.

Converts RDF URI triples into a hierarchical network. After a search, nodes and
edges on the GraphRAG walk are highlighted; the rest of the graph is dimmed.
The page is embedded in an iframe via srcdoc so Gradio can host the HTML.
"""
import json
from typing import List, Tuple
from rdflib import Graph, URIRef, RDF, RDFS


def generate_graph_html(
    kg_graph: Graph,
    active_nodes: List[str] = None,
    active_edges: List[Tuple[str, str, str]] = None
) -> str:
    """Return iframe HTML for the RDF graph, optionally highlighting a traversal path."""
    active_nodes_set = set(active_nodes or [])
    active_edges_set = {(str(s), str(p), str(o)) for s, p, o in (active_edges or [])}
    has_active_traversal = len(active_nodes_set) > 0

    # Node fill by ontology class (matches iMedical palette).
    color_map = {
        "ClinicalStudy": "#d51900",       # Red
        "DiseaseCondition": "#79529c",    # Purple
        "TherapeuticArea": "#ab47bc",
        "Treatment": "#1976d2",           # Blue (IMPTS)
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
        # Literals are properties on nodes, not graph edges in this view.
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
                        # Gold border + glow for traversed nodes; gray for the rest.
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

            # Treat an edge as active if listed, or if both endpoints were walked.
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
    # srcdoc iframe keeps Vis.js isolated from the Gradio parent page.
    return f'<iframe srcdoc="{escaped_html}" style="width: 100%; height: 620px; border: 1px solid #e0e0e0; border-radius: 8px;"></iframe>'
