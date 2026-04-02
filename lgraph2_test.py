from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
import os

# 1. Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    sentiment: str

# 2. Initialize the model
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

# 3. Define Nodes

# Let Claude analyze sentiment instead of simple keyword matching
def analyze_sentiment(state: State):
    last_message = state["messages"][-1].content
    
    response = llm.invoke([
        SystemMessage(content="Analyze the sentiment. Reply with ONLY one word: 'positive' or 'negative'."),
        HumanMessage(content=last_message)
    ])
    
    sentiment = response.content.strip().lower()
    sentiment = sentiment if sentiment in ["positive", "negative"] else "negative"
    return {"sentiment": sentiment}

def handle_positive(state: State):
    user_message = state["messages"][-1].content
    response = llm.invoke([
        SystemMessage(content="You are a cheerful assistant. Respond enthusiastically."),
        HumanMessage(content=user_message)
    ])
    return {"messages": [response]}

def handle_negative(state: State):
    user_message = state["messages"][-1].content
    response = llm.invoke([
        SystemMessage(content="You are an empathetic assistant. Respond with care and support."),
        HumanMessage(content=user_message)
    ])
    return {"messages": [response]}

# 4. Routing function
def route_by_sentiment(state: State) -> Literal["positive", "negative"]:
    return state["sentiment"]

# 5. Build Graph
builder = StateGraph(State)

builder.add_node("analyze", analyze_sentiment)
builder.add_node("positive", handle_positive)
builder.add_node("negative", handle_negative)

builder.add_edge(START, "analyze")

builder.add_conditional_edges(
    "analyze",
    route_by_sentiment,
    {
        "positive": "positive",
        "negative": "negative"
    }
)

builder.add_edge("positive", END)
builder.add_edge("negative", END)

# 6. Compile & Run
graph = builder.compile()

# Test it
print("--- Test 1 ---")
result = graph.invoke({"messages": [HumanMessage(content="I just got promoted today!")]})
print(f"Sentiment: {result['sentiment']}")
print(f"Response: {result['messages'][-1].content}")

print("\n--- Test 2 ---")
result = graph.invoke({"messages": [HumanMessage(content="I lost my keys and missed my meeting.")]})
print(f"Sentiment: {result['sentiment']}")
print(f"Response: {result['messages'][-1].content}")