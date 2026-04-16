import anthropic

client = anthropic.Anthropic()

session = client.beta.sessions.create(
    agent={"type": "agent", "id": "agent_011CZskMNKTpuGEJnmzrHfBh"},
    environment_id="env_01XzzScB8uywcvYh4otdK7Jt",
    betas=["managed-agents-2026-04-01"],
)

with client.beta.sessions.events.stream(session_id=session.id, betas=["managed-agents-2026-04-01"]) as stream:
    client.beta.sessions.events.send(
        session_id=session.id,
        events=[
            {
                "type": "user.message",
                "content": [{"type": "text", "text": "Can you advise on construction work on buildings?"}],
            },
        ],
        betas=["managed-agents-2026-04-01"],
    )

    for event in stream:
        if event.type == "agent.message":
            for block in event.content:
                print(block.text, end="")
        elif event.type == "agent.tool_use":
            print(f"\n[Using tool: {event.name}]")
        elif event.type == "session.status_idle":
            print("\n\nAgent finished.")
            break