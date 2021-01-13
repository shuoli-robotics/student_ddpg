from ksluck_sim_private.pybullet_ksluck.anymal_base import AnymalRobot
from ksluck_sim_private.pybullet_ksluck.anymal_base_env import AnymalBaseBulletEnv

class Anymal(AnymalBaseBulletEnv):
    def __init__(self, **kwargs):
        self.robot = AnymalRobot()
        super(Anymal, self).__init__(self.robot)
