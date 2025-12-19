import pandas as pd
import requests
import json

# -------------------------------
# CONFIG
# -------------------------------
LLM_API_URL = "https://api-int-hawkeye-dev.carbon.lowes.com/llama3/v1/chat/completions"
API_TOKEN = "PASTE_YOUR_TOKEN_HERE"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# -------------------------------
# LOAD EXCEL ONCE
# -------------------------------
df = pd.read_excel("data.xlsx")


# -------------------------------
# FIND RELEVANT ROWS (token-safe)
# -------------------------------
def get_relevant_rows(question, top_k=5):
    keywords = question.lower().split()
    mask = df.apply(
        lambda row: any(word in str(row).lower() for word in keywords),
        axis=1
    )
    return df[mask].head(top_k)


# -------------------------------
# CALL LLM (same as Postman)
# -------------------------------
def ask_excel_assistant(question):
    rows = get_relevant_rows(question)

    if rows.empty:
        context = "No directly matching rows found."
    else:
        context = rows.to_string(index=False)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are an intelligent Excel data assistant. Answer clearly and concisely."
            },
            {
                "role": "user",
                "content": f"""
Here is relevant data from the Excel sheet:

{context}

User question:
{question}

Answer in natural language.
"""
            }
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 300,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "user": "excel-assistant"
    }

    response = requests.post(
        LLM_API_URL,
        headers=HEADERS,
        data=json.dumps(payload),
        verify=False
    )

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# -------------------------------
# CHAT LOOP
# -------------------------------
if __name__ == "__main__":
    print("📊 Excel AI Assistant (type 'exit' to quit)\n")

    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break

        answer = ask_excel_assistant(question)
        print("\nBot:", answer, "\n")
