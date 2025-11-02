import requests

# Set your API token
api_token = "YOUR-API-TOKEN"

# Define the endpoint and parameters for Greece
endpoint = 'https://transparency.entsoe.eu/api'
params = {
    'documentType': 'A75',  # Actual Generation per Production Type
    'processType': 'A16',   # Realized data
    'area_Domain': '10YGR-HTSO-----Y',  # Country code for Greece
    'periodStart': '202401010000',  # Start date (format: YYYYMMDDHHMM)
    'periodEnd': '202401020000',    # End date (format: YYYYMMDDHHMM)
    'securityToken': api_token
}

# Make the request
response = requests.get(endpoint, params=params)

if response.status_code == 200:
    print("Data Retrieved Successfully")
    data = response.content
else:
    print("Failed to Retrieve Data", response.status_code)

with open('greece_generation_data.xml', 'wb') as file:
    file.write(data)
