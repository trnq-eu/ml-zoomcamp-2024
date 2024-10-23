import requests

url = "http://0.0.0.0:9696/predict"
client = {"job": "management", "duration": 400, "poutcome": "success"}
response = requests.post(url, json=client).json()

print(response)