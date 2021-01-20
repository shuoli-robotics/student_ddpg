from anymal_walking_envs.anymal_walking_env import AnymalWalkEnv as Env
import numpy as np
import matplotlib.pyplot as plt

env = Env()
states = env.reset([0.5,0.5,0])

print(env.observation_space)

for i in range(20):
    states = env.reset([0.5,0.5,0])
    temp = 1
    jointPositionCommand = []
    realJointPosition = []
    for j in range(2000):
        action = np.zeros(12)
        action[0] = 0.01
        jointPositionCommand.append(action[0])
        realJointPosition.append(env.robot.ordered_joints[0].current_position()[0])
        state, reward, terminal, info = env.step(action)
        temp = 1

    plt.figure(1)
    plt.plot(jointPositionCommand)
    plt.plot(realJointPosition)
    # plt.show()
    temp = 1



temp = 1

from Env.pybullet_adapted.gym_locomotion_envs import HalfCheetahBulletEnv as Env

