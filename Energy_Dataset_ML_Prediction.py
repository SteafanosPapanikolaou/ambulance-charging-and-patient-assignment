import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib

# Read the data
data = pd.read_csv('filepath')

# Convert timestamp column to datetime with specified format
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%d-%m-%y %H:%M')
data.set_index('timestamp', inplace=True)

# Extract date and time features
data['hour'] = data.index.hour
data['day_of_week'] = data.index.dayofweek
data['month'] = data.index.month
data['year'] = data.index.year

end_date = pd.to_datetime('2023-12-31 23:00:00')
RLdata = pd.DataFrame()
RLdata.index = data.index
RLdata['year'] = data['year']
RLdata['month'] = data['month']
RLdata['day_of_week'] = data['day_of_week']
RLdata['hour'] = data['hour']
RLdata = RLdata[RLdata.index>end_date]
EnergyIndex = ['Fossil Gas','Fossil Brown coal','Hydro Pumped Storage',
               'Hydro Water Reservoir','Wind Onshore','Solar']

# Load and use the models for predictions
for energy in EnergyIndex:
    features = ['hour', 'day_of_week', 'month', 'year', energy + ' trend',
                energy + ' seasonal_daily', energy + ' seasonal_yearly']
    file_path = f'filepath'

    # Load the model using joblib
    model = joblib.load(file_path)

    # Filter the prediction data (example: use data from a specific period)
    predict_data = data[data.index > end_date]
    predict_data = predict_data[features]

    # Check if all required features are in the dataset
    for feature in features:
        if feature not in data.columns:
            raise ValueError(f"Feature '{feature}' is not in the dataset.")

    # Perform prediction
    RLdata[energy] = model.predict(predict_data)

selected_data = []
# Upsample the data to minute frequency and forward fill the values
RLdata_minute = RLdata.resample('T').ffill()
TimePeriod = RLdata_minute.index.to_period('D').unique()
random_TP = np.random.choice(TimePeriod)
selected_data = RLdata_minute[(RLdata_minute.index >= random_TP.start_time) & (RLdata_minute.index <= random_TP.end_time)]
selected_data.insert(selected_data.columns.get_loc('hour') + 1, 'minute', selected_data.index.minute)