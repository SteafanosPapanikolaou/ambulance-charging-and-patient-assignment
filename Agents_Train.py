from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import gymnasium as gym
import os
import pandas as pd
from RL_Enviroment import EvAmbulanceEnv

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

metrics_list = []

# 🛠️ Create a folder for logs
log_dir = "./ppo_ev_logs/"
os.makedirs(log_dir, exist_ok=True)

# ✅ Wrap the custom env in a function
def make_env():
    env_temp = EvAmbulanceEnv(data)
    env = Monitor(env_temp)  # adds logging and episode reward tracking
    return env

# ✅ Wrap the custom env in a function
def make_env_second():
    env_temp  = EvAmbulanceEnv(data2, episode_length=8*1440)
    env = Monitor(env_temp)  # adds logging and episode reward tracking
    return env

# 🧪 Check env (optional)
check_env(make_env(), warn=True)

# 🎮 Vectorize env
env = DummyVecEnv([make_env])

# 🧪 Evaluation env for callback
eval_env = DummyVecEnv([make_env])

# 📋 Logging & evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=5_000,
    deterministic=True,
    render=False,
)

models = {
    "PPO": PPO("MlpPolicy", env, verbose=0),
    "PPO-Tuned": PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=1024, batch_size=64, verbose=0),
    "PPO-Long": PPO("MlpPolicy", env, n_steps=2048, batch_size=64, verbose=0),  # Long horizon
    "PPO-Small": PPO("MlpPolicy", env, n_steps=128, batch_size=32, verbose=0),  # Small horizon
    "A2C": A2C("MlpPolicy", env, verbose=0),
    "A2C-Tuned": A2C("MlpPolicy", env, learning_rate=7e-4, n_steps=10, gamma=0.95, verbose=0),
    "PPO-HighLR": PPO("MlpPolicy", env, learning_rate=1e-3, n_steps=1024, batch_size=64, verbose=0),  # Higher LR
    "PPO-ClipLow": PPO("MlpPolicy", env, clip_range=0.1, verbose=0),  # Small clip range
    "PPO-EntropyReg": PPO("MlpPolicy", env, ent_coef=0.05, verbose=0),  # Higher entropy bonus
    "A2C-GammaHigh": A2C("MlpPolicy", env, gamma=0.99, verbose=0)  # High discount factor
}

trained_models = {}
timesteps = {
    "PPO": 1000000,
    "PPO-Tuned": 1000000,
    "PPO-Long": 2*1000000,  # Long horizon
    "PPO-Small": 1000000/2,  # Small horizon
    "A2C": 2*1000000,
    "A2C-Tuned": 1000000/2,
    "PPO-HighLR": 1000000*0.75,  # Higher LR
    "PPO-ClipLow": 1000000,  # Small clip range
    "PPO-EntropyReg": 1000000,  # Higher entropy bonus
    "A2C-GammaHigh": 2*1000000  # High discount factor
    }

for name, model in models.items():
    print(f"🔁 Training {name}...")
    model.learn(total_timesteps=timesteps[name]/5)
    trained_models[name] = model

    # Evaluate the trained model
    test_env = make_env_second()
    obs, _ = test_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = test_env.step(action)

    metrics = test_env.get_metrics()
    metrics_list.append(metrics)

    model.save(f"filepath")
