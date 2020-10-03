import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={ 'radius_mean':3, 'texture_mean':1, 'perimeter_mean':3,
       'area_mean':4, 'smoothness_mean':5, 'compactness_mean':6, 'concavity_mean':7,
       'concave points_mean':9, 'symmetry_mean':8, 'fractal_dimension_mean':7,
       'radius_se':2, 'texture_se':3, 'perimeter_se':4, 'area_se':5, 'smoothness_se':2,
       'compactness_se':5, 'concavity_se':3, 'concave points_se':8, 'symmetry_se':7,
       'fractal_dimension_se':5, 'radius_worst':6, 'texture_worst':7,
       'perimeter_worst':6, 'area_worst':3, 'smoothness_worst':4,
       'compactness_worst':7, 'concavity_worst':8, 'concave points_worst':9,
       'symmetry_worst':3, 'fractal_dimension_worst':5})

print(r.json())


