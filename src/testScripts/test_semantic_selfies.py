import requests

# URL for the SELFIES semantic search endpoint
url = "http://localhost:8000//selfies_semantic_search_SELFormer"

# Example SELFIES strings (e.g. ethanol “[C][C][O]”)
payload = {
    "texts": ["[C][C][O]"],
    "top_k": 5
}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())
