from langfuse import Langfuse
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
import os

os.environ["ANTHROPIC_API_KEY"]   = "your-anthropic-key"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
os.environ["LANGFUSE_HOST"]       = "https://cloud.langfuse.com"

langfuse = Langfuse()
llm      = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# ─────────────────────────────────────────
# Manual Tracing
# ─────────────────────────────────────────
def answer_question(user_question: str, user_id: str = None):

    # Create a trace
    trace = langfuse.trace(
        name     = "question-answering",
        input    = {"question": user_question},
        user_id  = user_id,
        tags     = ["production", "qa"]
    )

    # Create a span for the LLM call
    span = trace.span(
        name  = "claude-call",
        input = {"question": user_question}
    )

    # Make the LLM call
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=user_question)
    ])

    answer = response.content

    # End the span with output
    span.end(output={"answer": answer})

    # Update trace with final output
    trace.update(output={"answer": answer})

    return answer, trace.id


# Run it
answer, trace_id = answer_question(
    user_question = "What is the capital of France?",
    user_id       = "user-123"
)

print(f"Answer: {answer}")
print(f"Trace ID: {trace_id}")

# Always flush at end of script
langfuse.flush()