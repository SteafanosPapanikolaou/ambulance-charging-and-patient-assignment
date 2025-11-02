import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/df_02_2024_distance_data.csv')
df_energy = pd.read_csv('/content/drive/MyDrive/df_02_2024_energy_predictions.csv')

df['index'] = df['minute']+df['hour']*60+(df['day']-1)*60*24
df.set_index('index', inplace=True)
merged_df = pd.merge(df_energy, df, left_index=True, right_index=True, how='left')
merged_df.drop(['year_x', 'month_x', 'day_of_week', 'hour_x',
                'year_y', 'month_y', 'day', 'hour_y', 'minute'], axis=1, inplace=True)
merged_df.head()
merged_df.to_csv('/content/drive/MyDrive/df_02_2024_merged_data.csv', index=True)
