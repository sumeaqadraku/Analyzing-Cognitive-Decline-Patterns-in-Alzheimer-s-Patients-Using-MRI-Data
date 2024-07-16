import requests

url = 'http://127.0.0.1:5000/predict'
data = {
    "age": 68,
    "education_years": 15,
    "mmse_score": 23,
    "moca_score": 20,
    "clock_drawing_score": 3
}
response = requests.post(url, json=data)
print(response.json())
