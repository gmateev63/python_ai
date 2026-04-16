from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
import os

os.environ["ANTHROPIC_API_KEY"]   = "your-anthropic-key"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
os.environ["LANGFUSE_HOST"]       = "https://cloud.langfuse.com"

langfuse = Langfuse()

# ─────────────────────────────────────────
# 1. Define State
# ─────────────────────────────────────────
class State(TypedDict):
    messages:  Annotated[list, add_messages]
    sentiment: str
    trace_id:  str

# ─────────────────────────────────────────
# 2. Define Nodes with Tracing
# ─────────────────────────────────────────
def analyze_sentiment(state: State):
    trace = langfuse.trace(name="sentiment-analysis")

    langfuse_handler = CallbackHandler(trace_id=trace.id)

    llm = ChatAnthropic(
        model     = "claude-3-5-sonnet-20241022",
        callbacks = [langfuse_handler]
    )

    last_message = state["messages"][-1].content

    response = llm.invoke([
        SystemMessage(content="Analyze sentiment. Reply ONLY with 'positive' or 'negative'."),
        HumanMessage(content=last_message)
    ])

    sentiment = response.content.strip().lower()
    sentiment = sentiment if sentiment in ["positive", "negative"] else "negative"

    trace.update(
        output = {"sentiment": sentiment},
        tags   = [sentiment]
    )

    return {"sentiment": sentiment, "trace_id": trace.id}


def handle_positive(state: State):
    langfuse_handler = CallbackHandler(
        trace_id = state.get("trace_id"),
        tags     = ["positive-handler"]
    )

    llm = ChatAnthropic(
        model     = "claude-3-5-sonnet-20241022",
        callbacks = [langfuse_handler]
    )

    response = llm.invoke([
        SystemMessage(content="You are cheerful. Respond enthusiastically!"),
        HumanMessage(content=state["messages"][-1].content)
    ])

    return {"messages": [response]}


def handle_negative(state: State):
    langfuse_handler = CallbackHandler(
        trace_id = state.get("trace_id"),
        tags     = ["negative-handler"]
    )

    llm = ChatAnthropic(
        model     = "claude-3-5-sonnet-20241022",
        callbacks = [langfuse_handler]
    )

    response = llm.invoke([
        SystemMessage(content="You are empathetic. Respond with care and support."),
        HumanMessage(content=state["messages"][-1].content)
    ])

    return {"messages": [response]}


# ─────────────────────────────────────────
# 3. Routing
# ─────────────────────────────────────────
def route_by_sentiment(state: State) -> Literal["positive", "negative"]:
    return state["sentiment"]


# ─────────────────────────────────────────
# 4. Build Graph
# ─────────────────────────────────────────
builder = StateGraph(State)

builder.add_node("analyze",  analyze_sentiment)
builder.add_node("positive", handle_positive)
builder.add_node("negative", handle_negative)

builder.add_edge(START, "analyze")

builder.add_conditional_edges(
    "analyze",
    route_by_sentiment,
    {"positive": "positive", "negative": "negative"}
)

builder.add_edge("positive", END)
builder.add_edge("negative", END)

graph = builder.compile()

# ─────────────────────────────────────────
# 5. Run with Tracing
# ─────────────────────────────────────────
test_messages = [
    "I just got promoted, best day ever!",
    "I failed my exam and feel terrible.",
]

for message in test_messages:
    print(f"\n{'='*50}")
    print(f"Input: {message}")

    result = graph.invoke({
        "messages": [HumanMessage(content=message)]
    })

    print(f"Sentiment: {result['sentiment']}")
    print(f"Response:  {result['messages'][-1].content}")
    print(f"Trace ID:  {result['trace_id']}")

langfuse.flush()