import matplotlib.pyplot as plt
import pandas as pd

def plot_training_rewards():
    data = pd.read_csv("logs/monitor.csv", skiprows=1)

    rewards = data["r"]
    plt.figure(figsize=(10,5))

    plt.plot(rewards)

    plt.xlabel("Episode")

    plt.ylabel("Reward")

    plt.title("Training Rewards")

    plt.grid()

    plt.savefig("logs/training_rewards.png")

    plt.close()