from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse.callback import CallbackHandler
import os

os.environ["ANTHROPIC_API_KEY"]   = "your-anthropic-key"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
os.environ["LANGFUSE_HOST"]       = "https://cloud.langfuse.com"

# ─────────────────────────────────────────
# Langfuse callback = automatic tracing!
# ─────────────────────────────────────────
langfuse_handler = CallbackHandler(
    user_id  = "user-123",
    session_id = "session-abc",
    tags     = ["production"]
)

llm = ChatAnthropic(
    model    = "claude-3-5-sonnet-20241022",
    callbacks = [langfuse_handler]   # 👈 Just add this!
)

# Every call is now automatically traced
response = llm.invoke([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Explain quantum computing in simple terms.")
])

print(response.content)