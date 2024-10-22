import requests

url = "http://0.0.0.0:9696/predict"
client = {"job": "student", "duration": 280, "poutcome": "failure"}
response = requests.post(url, json=client).json()

print(response)