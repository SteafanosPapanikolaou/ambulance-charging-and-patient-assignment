import pandas as pd
from statsmodels.tsa.seasonal import MSTL
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv('filepath')

# Convert timestamp column to datetime with specified format
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%d-%m-%y %H:%M')
data.set_index('timestamp', inplace=True)

# Generate a complete date range from the start to the end of your data
complete_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='h')

# Resampling to hourly frequency
data = data.resample('h').mean()

# Identify the missing hours by comparing the complete range with your data's index
missing_hours = complete_range.difference(data.index)

# Interpolate the missing values
data = data.interpolate(method='linear')

# Initialize a DataFrame to store the results
result_df = pd.DataFrame(index=data.index)

# Perform MSTL decomposition for each variable and save the results
for variable in data:
    # Apply MSTL decomposition
    stl = MSTL(data[variable], periods=[24, 8760])
    result = stl.fit()

    # Store the results in the result_df
    result_df[variable] = data[variable]
    result_df[variable + ' trend'] = result.trend
    result_df[variable + ' seasonal_daily'] = result.seasonal.iloc[:, 0].values
    result_df[variable + ' seasonal_yearly'] = result.seasonal.iloc[:, 1].values
    result_df[variable + ' residual'] = result.resid

    # Plot the results and save each plot as an image file
    fig = result.plot()
    plt.tight_layout()
    plt.savefig(f'{variable}_decomposition.png')
    plt.close(fig)

    print(variable)

# Save the result DataFrame to a new CSV file
result_df.to_csv('filepath')
