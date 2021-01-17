from anymal_walking_envs.anymal_walking_env import AnymalWalkEnv as Env
import numpy as np

env = Env()
states = env.reset()

print(env.observation_space)

for i in range(2):
    states = env.reset()
    temp = 1
    for j in range(20):
        action = np.random.rand(12)
        state, reward, terminal, info = env.step(action)





temp = 1

