from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from envs import GridWorldEnv


def train_model():

    env = GridWorldEnv()

    env = Monitor(env, filename="logs/monitor.csv")

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=250,
        exploration_fraction=0.9,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        verbose=0,
        tensorboard_log="./logs/"
    )

    model.learn(total_timesteps=1000000)

    model.save("models/gridworld_dqn")