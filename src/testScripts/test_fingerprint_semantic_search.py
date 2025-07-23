import requests

url = "http://localhost:8000/fingerprint_semantic_search"
payload = {
    "texts": ["CCO"],
    "top_k": 5
}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())
