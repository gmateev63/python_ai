from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
import os

# 1. Define the State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Initialize the model
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

# 3. Define nodes
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 4. Build the graph
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 5. Compile the graph
graph = graph_builder.compile()

# 6. Run it
result = graph.invoke({"messages": [("user", "What is 2 + 2?")]})
print(result["messages"][-1].content)