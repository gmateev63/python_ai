from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
import networkx as nx
import os

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

# ─────────────────────────────────────────
# 1. Build a Simple Knowledge Graph
# ─────────────────────────────────────────
def build_knowledge_graph():
    G = nx.Graph()

    # Add entities (nodes)
    entities = [
        ("Alice",   {"type": "Person", "role": "Engineer"}),
        ("Bob",     {"type": "Person", "role": "Manager"}),
        ("Charlie", {"type": "Person", "role": "Designer"}),
        ("ProjectX",{"type": "Project", "status": "Active"}),
        ("ProjectY",{"type": "Project", "status": "Completed"}),
        ("Python",  {"type": "Technology"}),
        ("React",   {"type": "Technology"}),
    ]
    G.add_nodes_from(entities)

    # Add relationships (edges)
    relationships = [
        ("Alice",    "ProjectX", {"relation": "works_on"}),
        ("Alice",    "Python",   {"relation": "uses"}),
        ("Bob",      "ProjectX", {"relation": "manages"}),
        ("Bob",      "ProjectY", {"relation": "managed"}),
        ("Charlie",  "ProjectX", {"relation": "works_on"}),
        ("Charlie",  "React",    {"relation": "uses"}),
        ("ProjectX", "Python",   {"relation": "requires"}),
        ("ProjectX", "React",    {"relation": "requires"}),
    ]
    G.add_edges_from(relationships)

    return G

knowledge_graph = build_knowledge_graph()

# ─────────────────────────────────────────
# 2. Define State
# ─────────────────────────────────────────
class State(TypedDict):
    messages:    Annotated[list, add_messages]
    query:       str
    subgraph:    str   # retrieved graph context
    answer:      str

# ─────────────────────────────────────────
# 3. Define Nodes
# ─────────────────────────────────────────

# Node 1: Extract entities from query
def extract_entities(state: State):
    query = state["messages"][-1].content

    response = llm.invoke([
        SystemMessage(content="""Extract entity names from the query.
        Only return comma-separated names from this list:
        Alice, Bob, Charlie, ProjectX, ProjectY, Python, React.
        If none found, return 'None'."""),
        HumanMessage(content=query)
    ])

    entities = [e.strip() for e in response.content.split(",")]
    return {"query": query, "entities": entities}


# Node 2: Retrieve subgraph
def retrieve_subgraph(state: State):
    query   = state["query"]
    G       = knowledge_graph
    context = []

    # Find relevant nodes based on query keywords
    relevant_nodes = [
        node for node in G.nodes()
        if node.lower() in query.lower()
    ]

    # If no direct match, search all nodes
    if not relevant_nodes:
        relevant_nodes = list(G.nodes())

    # Traverse graph to get relationships
    for node in relevant_nodes:
        node_data = G.nodes[node]
        context.append(f"\nEntity: {node} {dict(node_data)}")

        # Get all neighbors and their relationships
        for neighbor in G.neighbors(node):
            edge_data    = G.edges[node, neighbor]
            neighbor_data = G.nodes[neighbor]
            context.append(
                f"  → {node} --[{edge_data['relation']}]--> "
                f"{neighbor} {dict(neighbor_data)}"
            )

    subgraph_context = "\n".join(context)
    return {"subgraph": subgraph_context}


# Node 3: Generate answer from subgraph
def generate_answer(state: State):
    response = llm.invoke([
        SystemMessage(content=f"""You are a helpful assistant.
        Use the following Knowledge Graph context to answer the question.
        
        Knowledge Graph Context:
        {state['subgraph']}
        
        Answer based on the graph data provided."""),
        HumanMessage(content=state["query"])
    ])

    return {"answer": response.content, "messages": [response]}


# ─────────────────────────────────────────
# 4. Build the Graph
# ─────────────────────────────────────────
builder = StateGraph(State)

builder.add_node("extract",  extract_entities)
builder.add_node("retrieve", retrieve_subgraph)
builder.add_node("generate", generate_answer)

builder.add_edge(START,      "extract")
builder.add_edge("extract",  "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()

# ─────────────────────────────────────────
# 5. Run Queries
# ─────────────────────────────────────────
queries = [
    "Who is working on ProjectX?",
    "What technologies does Alice use?",
    "Who manages ProjectX and what is its status?",
    "What technologies are required for ProjectX?",
]

for query in queries:
    print(f"\n{'='*50}")
    print(f"Query: {query}")
    print('='*50)

    result = graph.invoke({
        "messages": [HumanMessage(content=query)]
    })

    print(f"Graph Context:\n{result['subgraph']}")
    print(f"\nAnswer: {result['answer']}")