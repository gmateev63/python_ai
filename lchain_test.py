from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001"
)

response = llm.invoke("Explain Redis in simple terms")

print(response.content)

