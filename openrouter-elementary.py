import requests
import json
from rich import print

OPEN_ROUTER_KEY = "..."

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": OPEN_ROUTER_KEY,
    "Content-Type": "application/json"
  },
  data=json.dumps({
    "model": "openai/gpt-oss-20b:free",
    "messages": [
      {
        "role": "user",
        "content": "Tell me more about KIA cars"
      }
    ]
  })
)

jsn = response.json()
mess = jsn['choices'][0]['message']['content']
print(mess)