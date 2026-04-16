from langfuse import Langfuse
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

langfuse = Langfuse()

# ─────────────────────────────────────────
# Push a prompt to Langfuse dashboard
# ─────────────────────────────────────────
langfuse.create_prompt(
    name    = "sentiment-analyzer",
    prompt  = "Analyze the sentiment of the text. Reply ONLY with 'positive' or 'negative'.",
    labels  = ["production"],
    config  = {"model": "claude-3-5-sonnet-20241022", "temperature": 0}
)

# ─────────────────────────────────────────
# Pull & use prompt from Langfuse
# ─────────────────────────────────────────
prompt_template = langfuse.get_prompt("sentiment-analyzer")

llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

response = llm.invoke([
    SystemMessage(content=prompt_template.prompt),  # fetched from Langfuse
    HumanMessage(content="I love this product!")
]) 

print(response.content)