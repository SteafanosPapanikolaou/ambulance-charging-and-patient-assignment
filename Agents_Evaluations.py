from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import gymnasium as gym
import os
from RL_Enviroment import EvAmbulanceEnv
import pandas as pd
import matplotlib as plt

model_dir = "filepath"
loaded_models = {}
model_names = ["PPO", "PPO-Tuned", "PPO-Long", "PPO-Small", "A2C", "A2C-Tuned",
               "PPO-HighLR", "PPO-ClipLow", "PPO-EntropyReg", "A2C-GammaHigh"]

for name in model_names:
    model_path = os.path.join(model_dir, f"{name}_model")

    # Choose correct class based on the name
    if name.startswith("PPO"):
        model = PPO.load(model_path)
    elif name.startswith("A2C"):
        model = A2C.load(model_path)
    else:
        raise ValueError(f"Unsupported model: {name}")

    loaded_models[name] = model
    print(f"✅ Loaded {name} model from {model_path}")

BatterySize = 68 #kWh for Ford E-Transit
EnergyConsum = 38.7 #kWh/100km for Ford E-Transit

df_merged_data = pd.read_csv("filepath")

data = pd.DataFrame()
data["calls"] = df_merged_data["emergency"].notna().astype(int)
data["emergency"] = df_merged_data["emergency"]
data["green"] = df_merged_data["Hydro Pumped Storage"] + df_merged_data["Hydro Water Reservoir"] + df_merged_data["Wind Onshore"] + df_merged_data["Solar"]
data["non_green"] = df_merged_data["Fossil Gas"] + df_merged_data["Fossil Brown coal"]
data["energy"] = data["green"] + data["non_green"]
data["green"] = data["green"] / data["energy"]
data["non_green"] = data["non_green"] / data["energy"]
data["unavailable_duration"] = round(df_merged_data["Travel_Time"]/60)
data["energy_required"] = df_merged_data["Length"]/1000*EnergyConsum/100/BatterySize
data["energy_required"] = data["energy_required"].fillna(0)
data["unavailable_duration"] = data["unavailable_duration"].fillna(0)
# data.head()
df = data

# Convert DataFrame to list of dicts for the env
data = df[["green", "non_green", "calls", "energy_required", "unavailable_duration", "emergency"]].to_dict(orient="records")

df2=df[20*1440:28*1440]
data2 = df2[["green", "non_green", "calls", "energy_required", "unavailable_duration", "emergency"]].to_dict(orient="records")

# 🧪 Evaluation env for callback
def make_env_second():
    env_temp  = EvAmbulanceEnv(data2, episode_length=8*1440)
    env = Monitor(env_temp)  # adds logging and episode reward tracking
    return env
def make_env():
    env_temp  = EvAmbulanceEnv(data)
    env = Monitor(env_temp)  # adds logging and episode reward tracking
    return env

eval_env = DummyVecEnv([make_env_second])
metrics_list = []
working_days = 8

for name, model in loaded_models.items():
  obs = eval_env.reset()
  rewards = []
  cumulative_rewards = []  # List to store cumulative rewards
  current_cumulative_reward = 0 # Initialize cumulative reward
  queue_sizes = []
  batteries = []
  emergencies = []

  raw_env = eval_env.envs[0].unwrapped  # 🔧 Unwrap the Monitor

  for step in range(0*1440,working_days*1440-1):  # visualize one episode
      action, _ = model.predict(obs)
      obs, reward, done, info = eval_env.step(action)

      rewards.append(reward[0])
      current_cumulative_reward += reward[0] # Add current step's reward to cumulative
      cumulative_rewards.append(current_cumulative_reward) # Append cumulative reward

      queue_sizes.append(len(raw_env.call_queue))
      batteries.append([
          amb["battery"] for amb in raw_env.ambulances
      ])
      emergencies.append(
          sum(1 for _, _, e, _, _ in raw_env.call_queue if e)
      )

      if done[0]:
          break

  metrics = raw_env.get_metrics()
  metrics_list.append(metrics)

  plt.figure(figsize=(12, 4)) # Increased figure size to accommodate new plot

  plt.subplot(1, 2, 1) # Changed to 3x2 subplot
  plt.plot(rewards)
  ymax = plt.ylim()[1]
  minutes_per_day = 1440
  plt.axvline(x=0, color='black', linestyle='--')
  for day in range(1,working_days+1):
    x_start = (day-1)*minutes_per_day
    x_end = day*minutes_per_day
    x_mid = x_start/2 + x_end/2
    plt.axvline(x=x_end, color='black', linestyle='--')
    plt.text(x_mid, ymax * 0.98, f'Day {day}', ha='center', va='top')

  plt.xlabel('Time Steps (min)')
  plt.ylabel('Reward per Step')
  plt.title(f"Reward per Step, {name}")

  plt.subplot(1, 2, 2) # Changed to 2x1 subplot
  plt.plot(cumulative_rewards) # Plot cumulative rewards.

  ymax = plt.ylim()[1]
  minutes_per_day = 1440
  plt.axvline(x=0, color='black', linestyle='--')
  for day in range(1,working_days+1):
    x_start = (day-1)*minutes_per_day
    x_end = day*minutes_per_day
    x_mid = x_start/2 + x_end/2
    plt.axvline(x=x_end, color='black', linestyle='--')
    plt.text(x_mid, ymax *(day/working_days)* 0.98, f'Day {day}', ha='center', va='top')

  plt.xlabel('Time Steps (min)')
  plt.ylabel('Cumulative Reward')
  plt.title(f"Cumulative Reward, {name}") # Title for cumulative reward plot

  plt.tight_layout()
  plt.show()

  plt.figure(figsize=(12, 4)) # Increased figure size to accommodate new plot

  for i in range(len(batteries[0])):
      plt.plot([b[i]*100 for b in batteries], label=f'Ambulance {i+1}')
  plt.legend()
  plt.xlabel('Time Steps (min)')
  plt.ylabel('Battery Percentage (%)')
  plt.title(f"Battery Levels, {name}")
  plt.tight_layout()
  plt.show()

  plt.figure(figsize=(12, 4)) # Increased figure size to accommodate new plot
  plt.subplot(2, 1, 1)
  plt.plot(queue_sizes)

  ymax = plt.ylim()[1]
  minutes_per_day = 1440
  plt.axvline(x=0, color='black', linestyle='--')
  for day in range(1,working_days+1):
    x_start = (day-1)*minutes_per_day
    x_end = day*minutes_per_day
    x_mid = x_start/2 + x_end/2
    plt.axvline(x=x_end, color='black', linestyle='--')
    plt.text(x_mid, ymax * 0.98, f'Day {day}', ha='center', va='top')

  plt.xlabel('Time Steps (min)')
  plt.ylabel('Max Calls in Queue')
  plt.title(f"Call Queue Size, {name}")

  b = pd.DataFrame(0,index=range(len(metrics['wait_data'])), columns=['Calls', 'Sum_Wait_time', 'Average_WT'])
  c = pd.DataFrame(0,index=range(len(metrics['wait_data'])), columns=['Calls', 'Sum_Wait_time', 'Average_WT'])
  for i in range(len(metrics['wait_data'])):
    starting = metrics['wait_data'].loc[i, 'Call_Time']
    ending = metrics['wait_data'].loc[i, 'Response_Time']
    if pd.isna(metrics['wait_data'].loc[i, 'Emergency_Type']):
      pass
    elif metrics['wait_data'].loc[i, 'Emergency_Type']=='True':

      k=0
      for j in range(int(starting),int(ending)+1):
        b.loc[j, 'Calls']+=1
        b.loc[j, 'Sum_Wait_time']+=k
        k += 1
    elif metrics['wait_data'].loc[i, 'Emergency_Type']=='False':

      k=0
      for j in range(int(starting),int(ending)+1):
        c.loc[j, 'Calls']+=1
        c.loc[j, 'Sum_Wait_time']+=k
        k += 1
  b['Average_WT'] = b['Sum_Wait_time']/b['Calls']
  c['Average_WT'] = c['Sum_Wait_time']/c['Calls']
  plt.subplot(2, 1, 2)
  b['Average_WT'] = b['Average_WT'].replace(np.nan, 0)
  window_size = 5
  b['average_wt_line'] = b['Average_WT'].rolling(window=2*window_size, center=True).mean()
  plt.plot(b['average_wt_line'])

  ymax = plt.ylim()[1]
  minutes_per_day = 1440
  plt.axvline(x=0, color='black', linestyle='--')
  for day in range(1,working_days+1):
    x_start = (day-1)*minutes_per_day
    x_end = day*minutes_per_day
    x_mid = x_start/2 + x_end/2
    plt.axvline(x=x_end, color='black', linestyle='--')
    plt.text(x_mid, ymax * 0.98, f'Day {day}', ha='center', va='top')

  plt.xlabel('Time Steps (min)')
  plt.ylabel('Average Waiting Time')
  plt.title(f"Average Waiting Time, {name}")

  plt.tight_layout()
  plt.show()