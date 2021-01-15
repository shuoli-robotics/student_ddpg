from ksluck_sim_private.pybullet_ksluck.anymal_envs import Anymal as Env
import numpy as np

env = Env()
states = env.reset()

print(env.observation_space)

for i in range(2):
    states = env.reset()
    temp = 1
    for j in range(200):
        action = np.random.rand(12)
        state, reward, terminal, info = env.step(action)





temp = 1

