import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"data": [1, 1, 1, 100, 200, 1, 2, 5.5, 10.0, 0.5, 0.5, 2.0, 0.0, 0.0, 2.5, 15.0]}
)
print(response.json())
