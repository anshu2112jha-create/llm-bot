import requests
import json
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# Disable SSL warnings (internal APIs)
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

# --------------------------------------------------
# CONFIG — COPY FROM POSTMAN
# --------------------------------------------------
CHAT_URL = "https://apis-b2b-devs.lowes.com/llama3/v1/chat/completions"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# 🔴 PASTE TOKEN FROM POSTMAN (WITHOUT 'Bearer ')
ACCESS_TOKEN = "PASTE_ACCESS_TOKEN_HERE"

# --------------------------------------------------
# SIMPLE LLM CALL
# --------------------------------------------------
def ask_llm(prompt):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 150,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "user": "manual-test"
    }

    response = requests.post(
        CHAT_URL,
        headers=headers,
        data=json.dumps(payload),
        verify=False
    )

    print("HTTP STATUS:", response.status_code)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print("ERROR RESPONSE:", response.text)
        return None


# --------------------------------------------------
# TEST
# --------------------------------------------------
if __name__ == "__main__":
    print("🧪 Simple LLM Test (type 'exit' to quit)\n")

    while True:
        q = input("You: ").strip()
        if q.lower() == "exit":
            break

        answer = ask_llm(q)
        if answer:
            print("Bot:", answer, "\n")
