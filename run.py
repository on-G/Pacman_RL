import os
from stable_baselines3 import DQN
from envs import GridWorldEnv
from train import train_model
from utils import visualize
import time

MODEL_PATH = "models/gridworld_dqn.zip"

if not os.path.exists(MODEL_PATH):
    print("Training model...")
    train_model()
    visualize.plot_training_rewards()
    print("Done.")

env = GridWorldEnv(render_mode="human")

model = DQN.load(MODEL_PATH)

for episode in range(5):
    print("Episode: ", episode)
    obs, info = env.reset()

    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        env.render()

        #time.sleep(0.1)

        done = terminated or truncated

env.close()

print("Finished")