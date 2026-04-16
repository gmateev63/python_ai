from langfuse import Langfuse

langfuse = Langfuse()

# ─────────────────────────────────────────
# Score a trace (e.g. after user feedback)
# ─────────────────────────────────────────
langfuse.score(
    trace_id = "your-trace-id",
    name     = "user-feedback",
    value    = 1,           # 1 = thumbs up, 0 = thumbs down
    comment  = "Very helpful response!"
)

# Numeric score
langfuse.score(
    trace_id = "your-trace-id",
    name     = "accuracy",
    value    = 0.95,
    comment  = "Almost perfect"
)

# ─────────────────────────────────────────
# Auto-evaluate with LLM
# ─────────────────────────────────────────
def auto_evaluate(trace_id: str, question: str, answer: str):
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

    eval_response = llm.invoke([
        SystemMessage(content="Rate the answer quality from 0.0 to 1.0. Reply with ONLY a number."),
        HumanMessage(content=f"Question: {question}\nAnswer: {answer}")
    ])

    score = float(eval_response.content.strip())

    langfuse.score(
        trace_id = trace_id,
        name     = "llm-eval-quality",
        value    = score
    )

    return score