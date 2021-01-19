import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
parentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0,parentdir)

from robot_bases import BodyPart
from robot_bases import XmlBasedRobot, MJCFBasedRobot, URDFBasedRobot
import numpy as np
import pybullet
import os
import pybullet_data

import sys, traceback


class AnymalRobot(URDFBasedRobot):

  def __init__(self):
    print("CREATE ROBOT")
    urdf_file_location = os.path.join(currentdir, os.path.join('data', os.path.join('anymal_flat', os.path.join('urdf', 'anymal_basic.urdf'))))
    print(urdf_file_location)
    self._robot_name = "anymal"
    self._possible_control_modes = [
        'pd_control', 'torque', 'velocity', 'position'
    ]
    self._control_mode = 'pd_control'
    action_dim = 12
    obs_dim = 34
    self_collision = False
    initial_height = 0.7
    URDFBasedRobot.__init__(self,
        model_urdf=urdf_file_location,
        robot_name=self._robot_name,
        action_dim=action_dim,
        obs_dim=obs_dim,
        basePosition=[0, 0, initial_height],
        baseOrientation=[0, 0, 0, 1],
        fixed_base=False,
        self_collision=self_collision)

    self.power = 1.0
    self.camera_x = 0
    self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
    self.walk_target_x = 1e3  # kilometer away
    self.walk_target_y = 0
    self.body_xyz = [0, 0, 0]

    self._action_mapping = {
                'LF_HAA': 0,  # Hip
                'LF_HFE': 1,      # Tigh
                'LF_KFE': 2,      # Shank
                'RF_HAA': 3,
                'RF_HFE': 4,
                'RF_KFE': 5,
                'LH_HAA': 6,
                'LH_HFE': 7,
                'LH_KFE': 8,
                'RH_HAA': 9,
                'RH_HFE': 10,
                'RH_KFE': 11,
            }
    self._initial_joint_pos = { # -0.03, -0.4, 0.8
                'LF_HAA': -0.03,  # Hip
                'LF_HFE': -0.4,      # Tigh
                'LF_KFE': 0.8,      # Shank
                'RF_HAA': -0.03,
                'RF_HFE': -0.4,
                'RF_KFE': 0.8,
                'LH_HAA': -0.03,
                'LH_HFE': -0.4,
                'LH_KFE': 0.8,
                'RH_HAA': -0.03,
                'RH_HFE': -0.4,
                'RH_KFE': 0.8,

            }

  def robot_specific_reset(self, bullet_client):
    self._p = bullet_client

    # Reset Joints here
    #self.scene.episode_restart(bullet_client)
    self._reset_joint_positions()
    self.scene.actor_introduce(self)

  def _reset_joint_positions(self):
        for j in self.ordered_joints:
            pos = self._initial_joint_pos[j.joint_name]
            j.reset_position(pos, 0.0)
            j.set_torque(0.0)


    # self.scene.actor_introduce(self)


  def apply_action(self, a):
      # TODO use the action mapping
    assert (np.isfinite(a).all())
    if self._control_mode == 'torque':
      for n, j in enumerate(self.ordered_joints):
          # This is for torque actions - change if we do position
          print(j.power_coef)
          j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))
    elif self._control_mode == 'pd_control':
      # motor_commands = self._compute_pd_control(a)
      for n, j in enumerate(self.ordered_joints):
          # This is for torque actions - change if we do position
          print(j.power_coef)
          j.set_pd_torque(float(a[n]), 40.0, 1.0)
    else:
        raise ValueError('No known control mode')


  def _compute_pd_control(self, a):
      motor_commands = np.array(a)
      print(motor_commands)
      state = self.calc_state()
      q = state['joint angles']
      print(q)
      qdot = state['joint velocities']
      print(qdot)
      kp = 40
      kd = 1

      torque_commands = -kp * (q - motor_commands) - kd * qdot
      print(torque_commands)

      joint_torque_limit = np.array([j.torque_limit for j in self.ordered_joints])
      torque_commands = np.clip( torque_commands, -joint_torque_limit, joint_torque_limit)

      return torque_commands

  def get_limits(self):
      joint_torque_limit = np.array([j.torque_limit for j in self.ordered_joints])

      return {
        'joint_torque_limit': joint_torque_limit,
      }


  def calc_state(self):
    """Convention

     observation space = [ height                                                    dim =  1
                         z-axis in world frame expressed in body frame (R_b.row(2))  dim =  3
                         joint angles,                                               dim = 12
                         body Linear velocities,                                     dim =  3
                         body Angular velocities,                                    dim =  3
                         joint velocities,                                           dim = 12] total dim 34
    """
    dummy_base = self.parts['base_inertia']
    base_position = self.parts[self._robot_name].current_position()
    height = base_position[2]
    base_orientation_quat = self.parts[self._robot_name].current_orientation()
    base_orientation_euler = self._p.getEulerFromQuaternion(base_orientation_quat)
    joint_positions = [j.get_position() for j in self.ordered_joints]
    joint_velocities = [j.get_velocity() for j in self.ordered_joints]
    base_velocity = self.parts['anymal'].speed()

    (x, y, z), (a, b, c, d), _, _, _, _, (vx, vy, vz), (vr, vp, vy) = self._p.getLinkState(
          dummy_base.bodies[dummy_base.bodyIndex], dummy_base.bodyPartIndex, computeLinkVelocity=1)
    body_angular_velocity = np.array([vr, vp, vz])

    return {
        'Height': height,
        'z-axis': base_orientation_euler,
        'joint angles': joint_positions,
        'body linear velocities' : base_velocity,
        'body angular velocities' : body_angular_velocity,
        'joint velocities': joint_velocities,

    }

  def calc_potential(self):
    # Just ignore this function
    return 0.0

  def _reset_joint_positions(self):
        for j in self.ordered_joints:
            pos = self._initial_joint_pos[j.joint_name]
            j.reset_position(pos, 0.0)
            j.set_torque(0.0)

  def set_initial_joint_positions(self, joint_dict):
        for name, val in joint_dict.items():
            self._initial_joint_pos[name] = val
