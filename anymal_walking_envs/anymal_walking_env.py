from ksluck_sim_private.pybullet_ksluck.anymal_base_env import AnymalBaseBulletEnv
from anymal_walking_envs.anymal_walking_robot import AnymalWalkRobot

class AnymalWalkEnv(AnymalBaseBulletEnv):
    def __init__(self, **kwargs):
        self.robot = AnymalWalkRobot()
        super(AnymalWalkEnv, self).__init__(self.robot)