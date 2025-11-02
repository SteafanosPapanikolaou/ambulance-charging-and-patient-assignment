import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense
# from scikeras.wrappers import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from tqdm.keras import TqdmCallback
import joblib
import os

# Ensure plot directory exists
plot_dir = 'plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

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

def SplitTimeSeriesYearly(data, end_date):
    train_data = data.loc[:end_date]
    end_test = end_date + pd.DateOffset(years=1)
    test_data = data.loc[end_date:end_test]
    return train_data, test_data

def TrainTimeSeriesYearlyModel(model, target, features, data, end_date, model_type='sklearn'):
    train_data, test_data = SplitTimeSeriesYearly(data, end_date)

    if model_type in ['LSTM', 'RNN']:
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        train_scaled_X = scaler_X.fit_transform(train_data[features])
        train_scaled_y = scaler_y.fit_transform(train_data[[target]])
        test_scaled_X = scaler_X.transform(test_data[features])
        test_scaled_y = scaler_y.transform(test_data[[target]])

        X_train, y_train = train_scaled_X, train_scaled_y
        X_test, y_test = test_scaled_X, test_scaled_y

        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        model.fit(X_train, y_train, epochs=50, batch_size=72, validation_data=(X_test, y_test), verbose=0, shuffle=False, callbacks=[TqdmCallback(verbose=1)])
        predicts = model.predict(X_test)

        y_test = scaler_y.inverse_transform(y_test)
        predicts = scaler_y.inverse_transform(predicts)

        # Reshape y_test and predicts to be 1D arrays for RMSE calculation
        y_test = y_test.ravel()
        predicts = predicts.ravel()

    else:
        model.fit(train_data[features], train_data[target])
        predict_date = end_date + pd.DateOffset(weeks=8)
        test = data.loc[end_date:predict_date]
        predicts = model.predict(test[features])
        y_test = test[target]

    rmse = np.sqrt(mean_squared_error(y_test, predicts))
    print(f'RMSE: {rmse}')

    # Save the plot
    plt.figure(figsize=(14, 7))
    plt.plot(test_data.index[:len(y_test)], y_test, label='Actual')
    plt.plot(test_data.index[:len(predicts)], predicts, label='Predicted')
    plt.legend()
    plt.title(f'Actual vs Predicted {target} Energy Production')
    plt.xlabel('Time')
    plt.ylabel('Energy Produced')
    plt.savefig(os.path.join(plot_dir, f'best_model_{target}.png'))
    plt.close()

    return rmse, model

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model

def create_rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(50, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model

# Define target and features
features = ['hour', 'day_of_week', 'month', 'year', 'Wind Onshore trend', 'Wind Onshore seasonal_daily', 'Wind Onshore seasonal_yearly']

# Define the initial end_date
end_date = pd.to_datetime('2023-12-31 23:00:00')

# Read the data
data21 = pd.read_csv('filepath')

# Dictionary to hold best models and their RMSE for each target
best_models = {}
best_rmse_dict = {}
# data21=['Fossil Gas','Fossil Brown coal','Hydro Pumped Storage','Hydro Water Reservoir','Wind Onshore','Solar']
data21=['Hydro Pumped Storage','Hydro Water Reservoir','Wind Onshore','Solar']
for target in data21:
    if target != 'timestamp':
        # Define the parameter grid for RandomForest
        rf_param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        features = ['hour', 'day_of_week', 'month', 'year', target+' trend', target+' seasonal_daily', target+' seasonal_yearly']
        rf_model = RandomForestRegressor()
        rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, scoring='neg_mean_squared_error', cv=3)
        rf_rmse, best_rf_model = TrainTimeSeriesYearlyModel(rf_grid_search, target, features, data, end_date)

        xgb_param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7]
        }
        xgb_model = xgb.XGBRegressor()
        xgb_grid_search = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, scoring='neg_mean_squared_error', cv=3)
        xgb_rmse, best_xgb_model = TrainTimeSeriesYearlyModel(xgb_grid_search, target, features, data, end_date)

        # Grid search for LSTM
        lstm_param_grid = {
            'batch_size': [32, 64],
            'epochs': [50],
            'units': [50, 100]
        }

        best_lstm_rmse = float('inf')
        best_lstm_model = None

        for batch_size in lstm_param_grid['batch_size']:
            for units in lstm_param_grid['units']:
                def create_lstm_model_grid(input_shape):
                    model = Sequential()
                    model.add(LSTM(units, input_shape=input_shape))
                    model.add(Dense(1))
                    model.compile(loss='mae', optimizer='adam')
                    return model

                lstm_model = KerasRegressor(model=create_lstm_model_grid, input_shape=(1, len(features)), epochs=50, batch_size=batch_size, verbose=0)
                lstm_rmse, trained_lstm_model = TrainTimeSeriesYearlyModel(lstm_model, target, features, data, end_date, model_type='LSTM')

                if lstm_rmse < best_lstm_rmse:
                    best_lstm_rmse = lstm_rmse
                    best_lstm_model = trained_lstm_model

        # Grid search for RNN
        rnn_param_grid = {
            'batch_size': [32, 64],
            'epochs': [50],
            'units': [50, 100]
        }

        best_rnn_rmse = float('inf')
        best_rnn_model = None

        for batch_size in rnn_param_grid['batch_size']:
            for units in rnn_param_grid['units']:
                def create_rnn_model_grid(input_shape):
                    model = Sequential()
                    model.add(SimpleRNN(units, input_shape=input_shape))
                    model.add(Dense(1))
                    model.compile(loss='mae', optimizer='adam')
                    return model

                rnn_model = KerasRegressor(model=create_rnn_model_grid, input_shape=(1, len(features)), epochs=50, batch_size=batch_size, verbose=0)
                rnn_rmse, trained_rnn_model = TrainTimeSeriesYearlyModel(rnn_model, target, features, data, end_date, model_type='RNN')

                if rnn_rmse < best_rnn_rmse:
                    best_rnn_rmse = rnn_rmse
                    best_rnn_model = trained_rnn_model

        # Select the best model
        # best_rmse, best_model = min([(rf_rmse, best_rf_model), (xgb_rmse, best_xgb_model), (best_lstm_rmse, best_lstm_model), (best_rnn_rmse, best_rnn_model)], key=lambda x: x[0])
        best_rmse, best_model = min([(rf_rmse, best_rf_model)], key=lambda x: x[0])
        best_models[target] = best_model
        best_rmse_dict[target] = best_rmse

        print(f'Best model for {target} with RMSE: {best_rmse}')

        # Save the best model using joblib
        joblib.dump(best_model, f'filepath')

# Save RMSE values to a CSV
rmse_df = pd.DataFrame.from_dict(best_rmse_dict, orient='index', columns=['RMSE'])
rmse_df.to_csv('filepath')

print("Training completed. Best models and RMSE values have been saved.")
