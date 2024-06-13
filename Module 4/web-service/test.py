import requests

ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40.0,
}

response = requests.post('http://localhost:9696/predict', json=ride).json()
print(response.json())