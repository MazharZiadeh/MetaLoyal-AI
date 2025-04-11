from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from metaloyal.envs.temptation_bridge import TemptationBridgeEnv
from config import CONFIG
import os

os.makedirs("results", exist_ok=True)

env = DummyVecEnv([lambda: TemptationBridgeEnv(
    alpha=CONFIG["alpha"],
    beta=CONFIG["beta"],
    lambda_loyalty=CONFIG["lambda_loyalty"],
    max_steps=CONFIG["max_steps"]
)])

model = PPO.load(CONFIG["model_path"])

obs = env.reset()
for _ in range(CONFIG["max_steps"]):
    action, _states = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    if done:
        break

trained_env = env.envs[0]

with open("results/reward_log.txt", "w") as f:
    f.write(",".join(map(str, trained_env.total_reward_log)))

with open("results/loyalty_log.txt", "w") as f:
    f.write(",".join(map(str, trained_env.loyalty_log)))

with open("results/betrayal_log.txt", "w") as f:
    f.write(",".join(map(str, trained_env.betrayals_log)))

print("✅ Evaluation logs saved in /results")
