import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={ 'radius_mean':17.99, 'texture_mean':10.38})

print(r.json())


