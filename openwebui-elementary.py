import requests, json
#from rich import print

URL = "https://lab.rila.bg/api/chat/completions"
'''
OPENAI_API_KEY = None          # optional – used by openWebUI behind the scenes
headers = {"Content-Type": "application/json"}
headers.update({"Authorization": f"Bearer {OPENAI_API_KEY}"} if OPENAI_API_KEY else {})
'''
data = {
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is quantum computing?"}
  ],
  "temperature": 0.7,
  "max_tokens": 150,
  "stream": False,
}
print(requests.post(URL, json=data, timeout=120).json()["messages"][-1]["content"])