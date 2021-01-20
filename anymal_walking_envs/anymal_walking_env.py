from ksluck_sim_private.pybullet_ksluck.anymal_base_env import AnymalBaseBulletEnv
from anymal_walking_envs.anymal_walking_robot import AnymalWalkRobot
import pybullet
from pybullet_envs.env_bases import MJCFBaseBulletEnv

class AnymalWalkEnv(AnymalBaseBulletEnv):
    def __init__(self, **kwargs):
        self.robot = AnymalWalkRobot()
        super(AnymalWalkEnv, self).__init__(self.robot)

    # check the safety of overriding and call parent's method !!
    def reset(self,robot_cmd):
        super(AnymalWalkEnv, self).reset()
        self.robot.cmd = robot_cmd

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit
        done = self._isDone()
        rest_states = self.robot.calc_rest_states_for_reward()

        reward = 1.0/ abs(state[1]-1.0)

        return state, reward, bool(done), {}