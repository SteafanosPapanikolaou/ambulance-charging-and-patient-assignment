import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class EvAmbulanceEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data, max_calls=10, episode_length=20*1440, n_ambulances=6):
        self.data = data
        self.max_calls = max_calls
        self.episode_length = episode_length
        self.n_ambulances = n_ambulances
        self.max_battery = 1.0

        #Tracking
        self.green_energy_used = pd.DataFrame(np.nan, index=range(episode_length),
                                                columns=[f'col_{i+1}' for i in range(n_ambulances)], dtype=object)
        self.non_green_energy_used = pd.DataFrame(np.nan, index=range(episode_length),
                                                columns=[f'col_{i+1}' for i in range(n_ambulances)], dtype=object)
        self.discharge_green_energy_given_back = pd.DataFrame(np.nan, index=range(episode_length),
                                                columns=[f'col_{i+1}' for i in range(n_ambulances)], dtype=object)
        self.discharge_energy_given_back = pd.DataFrame(np.nan, index=range(episode_length),
                                                columns=[f'col_{i+1}' for i in range(n_ambulances)], dtype=object)
        self.wait_times = pd.DataFrame({
            'Emergency_Type': pd.Series([np.nan]*episode_length, dtype='string'),
            'Call_Time': pd.Series([np.nan]*episode_length, dtype='float'),
            'Response_Time': pd.Series([np.nan]*episode_length, dtype='float')
        })

        self.action_space = spaces.MultiDiscrete([4] * self.n_ambulances)  # same: 0 = wait, 1 = charge, 2 = dispatch, 3 = discharge

        # 🆕 New observation: battery per ambulance, calls, time, etc.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3 * n_ambulances + 3,),  # e.g., battery, busy, mission time for each
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
      super().reset(seed=seed)
      self.time = 0
      self.call_queue = []

      self.ambulances = [{
          "battery": self.max_battery,
          "mission_time": 0,
          "charge_rate": 0.02,
      } for _ in range(self.n_ambulances)]

      return self._get_obs(), {}

    def step(self, actions):
      reward = 0
      done = False
      info = {}

      row = self.data[self.time]
      green = row["green"]
      new_calls = row["calls"]
      energy_required = row["energy_required"]
      unavailable_time = row["unavailable_duration"]
      is_emergency = row["emergency"]

      # Enqueue new calls
      for _ in range(new_calls):
        emergency = row.get("emergency", False)  # from your data
        self.call_queue.append((energy_required, unavailable_time, emergency, is_emergency, self.time))

      # Update ambulances
      for amb in self.ambulances:
          if amb["mission_time"] > 0:
              amb["mission_time"] -= 1
              if amb["mission_time"] == 0:
                  reward += 1  # mission complete bonus

      # Apply action to the **first available ambulance**
      for i, amb in enumerate(self.ambulances):
          action = actions[i]

          # Update mission timers
          if amb["mission_time"] > 0:
              self.green_energy_used.at[self.time, f'col_{i+1}'] = 'Mission in Progress'
              self.non_green_energy_used.at[self.time, f'col_{i+1}'] = 'Mission in Progress'
              self.discharge_green_energy_given_back.at[self.time, f'col_{i+1}'] = 'Mission in Progress'
              self.discharge_energy_given_back.at[self.time, f'col_{i+1}'] = 'Mission in Progress'
              amb["mission_time"] -= 1
              if amb["mission_time"] == 0:
                  reward += 1  # reward for finishing mission
              continue

          # If ambulance is available
          if action == 1:  # charge
              amb["battery"] = min(self.max_battery, amb["battery"] + amb["charge_rate"])
              if green > 0.6:
                reward += 5
              elif green > 0.5:
                reward += 3
              else:
                reward -= 1

              if green > 0.5:
                self.green_energy_used.at[self.time, f'col_{i+1}'] = amb["charge_rate"]
              else:
                self.non_green_energy_used.at[self.time, f'col_{i+1}'] = amb["charge_rate"]

          elif action == 3:  # discharge
              amb["battery"] = max(0, amb["battery"] - amb["charge_rate"])
              if amb["battery"] > 0.8:
                if green < 0.3:
                  reward += 4
                elif green < 0.4:
                  reward += 3
                elif green < 0.5:
                  reward += 1
                else:
                  reward -= 1
              elif amb["battery"] > 0.5:
                if green < 0.3:
                  reward += 3
                elif green < 0.4:
                  reward += 2
                elif green < 0.5:
                  reward += 1
                else:
                  reward -= 1
              else:
                reward -= 2

              if green > 0.5:
                self.discharge_green_energy_given_back.at[self.time, f'col_{i+1}'] = amb["charge_rate"]
              else:
                self.discharge_energy_given_back.at[self.time, f'col_{i+1}'] = amb["charge_rate"]

          elif action == 2 and self.call_queue:
              energy, duration, emergency, is_emergency, call_time = self.call_queue[0]
              if amb["battery"] >= energy:
                  amb["battery"] -= energy
                  amb["mission_time"] = duration
                  self.wait_times.at[self.time, 'Call_Time'] = call_time
                  self.wait_times.at[self.time, 'Response_Time'] = self.time
                  self.wait_times.at[self.time, 'Emergency_Type'] = str(is_emergency)
                  self.call_queue.pop(0)

                  reward += 5
                  if green > 0.5:
                      reward -= 1  # dispatch during peak green = opportunity loss
              else:
                if is_emergency == 'True':
                  reward -= 10  # tried to dispatch but lacked energy # high penalty for unhandled emergency
                else:
                  reward -= 5  # tried to dispatch but lacked energy # low penalty for non-emergency

      # Penalty for delayed emergency calls
      for _, _, emergency, is_emergency, _ in self.call_queue:
          if emergency == 'True':
            if is_emergency == 'True':
              reward -= 10  # high penalty for unhandled emergency
            else:
              reward -= 5  # low penalty for non-emergency
          else:
              reward -= 1  # low penalty for non-emergency

      self.time += 1
      done = self.time >= self.episode_length
      obs = self._get_obs()
      return obs, reward, done, False, info

    def _get_obs(self):
      row = self.data[self.time]
      obs = []

      num_emergencies = sum(1 for _, _, e, _, _ in self.call_queue if e)
      num_non_emergencies = len(self.call_queue) - num_emergencies

      obs = []
      for amb in self.ambulances:
          obs.append(amb["battery"])
          obs.append(1.0 if amb["mission_time"] > 0 else 0.0)
          obs.append(amb["mission_time"] / 20.0)

      obs.append(num_emergencies / self.max_calls)
      obs.append(num_non_emergencies / self.max_calls)
      obs.append(self.time / self.episode_length)

      return np.array(obs, dtype=np.float32)

    def get_metrics(self):
        green_energy_used = self.green_energy_used
        non_green_energy_used = self.non_green_energy_used
        discharge_green_energy_given_back = self.discharge_green_energy_given_back
        discharge_energy_given_back = self.discharge_energy_given_back
        wait_data = self.wait_times

        return {
            "green_energy_used": green_energy_used,
            "non_green_energy_used": non_green_energy_used,
            "discharge_green_energy_given_back": discharge_green_energy_given_back,
            "discharge_energy_given_back": discharge_energy_given_back,
            "wait_data": wait_data,
        }

    def render(self):
        print(f"[T={self.time}] Battery: {self.battery:.2f} | Calls: {self.call_queue}")