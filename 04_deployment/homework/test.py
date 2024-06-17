import json
from pprint import pprint

import requests

# url = 'http://127.0.0.1:9696/predict'
url = "http://0.0.0.0:9696/predict"
data = {
    "year": 2023,
    "month": 5
}
response = requests.post(url, json=data)
print(response)
pprint(json.loads(response.content))