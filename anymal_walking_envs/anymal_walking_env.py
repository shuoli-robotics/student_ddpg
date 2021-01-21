from ksluck_sim_private.pybullet_ksluck.anymal_base_env import AnymalBaseBulletEnv
from anymal_walking_envs.anymal_walking_robot import AnymalWalkRobot
import pybullet
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from math import exp
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R

class AnymalWalkEnv(AnymalBaseBulletEnv):
    def __init__(self, **kwargs):
        self.robot = AnymalWalkRobot()  # todo: is self necessary?
        super(AnymalWalkEnv, self).__init__(self.robot)

        self.step_counter = 0
        self.k_c = 0.3
        self.k_d = 0.997

    # check the safety of overriding and call parent's method !!
    def reset(self,robot_cmd):
        super(AnymalWalkEnv, self).reset()
        self.step_counter = 0
        self.k_c = 0.3
        self.robot.cmd = robot_cmd

    def step(self, a):
        self.step_counter += 1
        self.robot.apply_action(a)
        self.scene.global_step()
        obs,state_dict = self.robot.calc_state()  # also calculates self.joints_at_limit
        done = self._isDone()

        reward = self.calc_reward(state_dict)

        temp = 1

        return obs, reward, bool(done), {}

    def calc_reward(self,state_dict):
        self.k_c = self.k_c ** self.k_d

        deltaT = self.scene.timestep

        c_w = -6*deltaT
        c_v1 = -10 * deltaT
        c_v2 = -4 * deltaT
        c_tau = 0.005*deltaT
        c_js = 0.03*deltaT
        c_f = 0.1*deltaT
        c_fv = 2.0* deltaT
        c_o = 0.4* deltaT
        c_s = 0.5*deltaT

        rate_cost = c_w*self.logistic_kernel(abs(state_dict['vel_cmd'][2]-state_dict['rate'][2]))

        vel_cost = c_v1 * self.logistic_kernel( LA.norm(c_v2*(state_dict['vel_cmd'][0:2] - state_dict['vel'][0:2])))

        torque_cost = self.k_c* c_tau * (LA.norm(state_dict['joint_torque']))**2

        joint_speed_cost = self.k_c*c_js * LA.norm(state_dict['joint_vel'])**2

        clearance_cost = self.k_c * c_f * (
                abs(state_dict['foot_pos_z'][0] - 0.07) * state_dict['foot_vel'][0] +
                abs(state_dict['foot_pos_z'][1] - 0.07) * state_dict['foot_vel'][1] +
                abs(state_dict['foot_pos_z'][2] - 0.07) * state_dict['foot_vel'][2] +
                abs(state_dict['foot_pos_z'][3] - 0.07) * state_dict['foot_vel'][3]
        )

        foot_slip_cost = self.k_c * c_fv * LA.norm(state_dict['foot_vel'][0])

        r = R.from_euler('zyx', [0.0,state_dict['foot_vel'][1],state_dict['foot_vel'][2]], degrees=False)
        r_vec = r.as_rotvec()
        r_vec_unit = r_vec / LA.norm(r_vec)
        orientation_cost = self.k_c*c_o * LA.norm([0,0,-1] - r_vec_unit)

        smoothness_cost = self.k_c * c_s * LA.norm(state_dict['delta_joint_torque'])

        return rate_cost+vel_cost+torque_cost+joint_speed_cost+clearance_cost+foot_slip_cost+orientation_cost+smoothness_cost

    def logistic_kernel(self,x):
        return -1.0 / (exp(x) + 2 + exp(-x))

