import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from metaloyal.envs.temptation_bridge import TemptationBridgeEnv
from config import CONFIG

def make_env():
    return TemptationBridgeEnv(
        alpha=CONFIG["alpha"],
        beta=CONFIG["beta"],
        lambda_loyalty=CONFIG["lambda_loyalty"],
        max_steps=CONFIG["max_steps"]
    )

env = DummyVecEnv([make_env])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=CONFIG["total_timesteps"])

os.makedirs("models", exist_ok=True)
model.save(CONFIG["model_path"])

print(f"✅ Model saved to {CONFIG['model_path']}")
