import requests

# Use the correct port here
url = "http://127.0.0.1:5001/predict/"

data = {
    "Time": 1000.0,
    "V1": 0.5,
    "V2": 1.2,
    "V3": 2.3,
    "V4": 3.4,
    "V5": 4.5,
    "V6": 5.6,
    "V7": 6.7,
    "V8": 7.8,
    "V9": 8.9,
    "V10": 9.1,
    "V11": 10.2,
    "V12": 11.3,
    "V13": 12.4,
    "V14": 13.5,
    "V15": 14.6,
    "V16": 15.7,
    "V17": 16.8,
    "V18": 17.9,
    "V19": 18.0,
    "V20": 19.1,
    "V21": 20.2,
    "V22": 21.3,
    "V23": 22.4,
    "V24": 23.5,
    "V25": 24.6,
    "V26": 25.7,
    "V27": 26.8,
    "V28": 27.9,
    "Amount": 100.00
}

response = requests.post(url, json=data)

print("Status code:", response.status_code)
try:
    print("Response JSON:", response.json())
except ValueError:
    print("Response content is not valid JSON:", response.text)
