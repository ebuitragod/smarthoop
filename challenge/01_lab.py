import gym
from gym import wrappers, logger
from datetime import datetime

from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN


# Custom MLP policy of two layers of size 32 each
class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[32, 32],
                                           layer_norm=False,
                                           feature_extraction="mlp")

env = gym.make('LunarLanderSmarthoop-v0')
env = DummyVecEnv([lambda: env])
env.seed(0)


model = DQN(CustomDQNPolicy, env, verbose=1)
# Train the agent
model.learn(total_timesteps=1000000)
now = datetime.now()
now_str = now.strftime("%Y-%m-%d")
model.save(now_str + "_LunarLanderSmarthoopLab")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()