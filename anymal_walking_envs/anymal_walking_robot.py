from ksluck_sim_private.pybullet_ksluck.anymal_base import AnymalRobot
from ksluck_sim_private.pybullet_ksluck.robot_bases import URDFBasedRobot
import numpy as np
import os,inspect
import pybullet_data
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as R

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
parentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0,parentdir)
dirname = os.path.dirname(__file__)

class AnymalWalkRobot(AnymalRobot):
    def __init__(self):
        print("CREATE ROBOT")

        # urdf_file_location = os.path.join(currentdir, os.path.join('data', os.path.join('anymal_flat',
        #                                                                                 os.path.join('urdf',
        #                                                                                              'anymal_basic.urdf'))))
        urdf_file_location = os.path.join(dirname, '../ksluck_sim_private/pybullet_ksluck/data/anymal_flat/urdf/anymal_basic.urdf')

        print(urdf_file_location)
        self._robot_name = "anymal"
        self._possible_control_modes = [
            'pd_control', 'torque', 'velocity', 'position'
        ]
        self._control_mode = 'pd_control'
        action_dim = 8
        obs_dim = 41
        self_collision = False
        initial_height = 0.6
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
            'LF_HFE': 1,  # Tigh
            'LF_KFE': 2,  # Shank
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
        self._initial_joint_pos = {  # -0.03, -0.4, 0.8
            'LF_HAA': -0.03,  # Hip
            'LF_HFE': -0.4,  # Tigh
            'LF_KFE': 0.8,  # Shank
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

        self.joint_pos_history = np.zeros((5000,12))
        self.joint_vel_history = np.zeros((5000,12))
        self.action_history = np.zeros((5000,8))
        self.joint_torque_history = np.zeros((5000, 12))
        self.joint_history_pointer = -1
        self.action_history_pointer = -1
        self.joint_torque_history_pointer = -1
        self.history_steps_1 = 20
        self.history_steps_2 = self.history_steps_1 * 2
        self.cmd = [1, 0.5, 0]

    def reset(self, bullet_client):
        s = super(AnymalRobot,self).reset(bullet_client)
        self.joint_pos_history = np.zeros((5000,12))
        self.joint_vel_history = np.zeros((5000,12))
        self.action_history = np.zeros((5000,8))
        self.joint_torque_history = np.zeros((5000, 12))
        self.joint_history_pointer = -1
        self.action_history_pointer = -1
        self.joint_torque_history_pointer = -1
        return s

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
            revolute_joint_counter = 0
            for n, j in enumerate(self.ordered_joints):
                # joint_pos = self.convert2pos(a[n],j.upperLimit,j.lowerLimit)
                # joint_pos = a[n]
                # Tune joint position controller's PD here
                if j.joint_name[3:] != 'KFE':
                    joint_pos = self.convert2pos(a[revolute_joint_counter],j.upperLimit,j.lowerLimit)
                    revolute_joint_counter += 1
                    j.set_pd_torque(joint_pos, 0.1, 0.5)
        else:
            raise ValueError('No known control mode')

        self.action_history_pointer = self.action_history_pointer + 1
        self.action_history[self.action_history_pointer] = a

    def convert2pos(self,action,joint_max_pos,joint_min_pos):
        dis2min = action + 1
        prop = dis2min / 2.0
        return prop * (joint_max_pos-joint_min_pos) + joint_min_pos

    def calc_state(self):
        dummy_base = self.parts['base_inertia']
        base_position = self.parts[self._robot_name].current_position()
        height = base_position[2]
        base_orientation_quat = self.parts[self._robot_name].current_orientation()
        base_orientation_euler = self._p.getEulerFromQuaternion(base_orientation_quat)
        joint_positions = [j.get_position() for j in self.ordered_joints]
        joint_velocities = [j.get_velocity() for j in self.ordered_joints]

        r = R.from_euler('zyx', [0.0, base_orientation_euler[1], base_orientation_euler[2]], degrees=False)
        gravity_b = r.apply([0, 0, -1])


        # Handle joints history
        self.joint_history_pointer = self.joint_history_pointer + 1
        self.joint_pos_history[self.joint_history_pointer] = joint_positions
        self.joint_vel_history[self.joint_history_pointer] = joint_velocities

        if self.joint_history_pointer <  self.history_steps_1:
            joint_pos_at_1 = [0.0] * 12
            joint_pos_at_2 = [0.0] * 12
            joint_vel_at_1 = [0.0] * 12
            joint_vel_at_2 = [0.0] * 12
        elif self.joint_history_pointer >  self.history_steps_1 and self.joint_history_pointer <  self.history_steps_2:
            joint_pos_at_1 = self.joint_pos_history[self.joint_history_pointer - self.history_steps_1]
            joint_vel_at_1 = self.joint_vel_history[self.joint_history_pointer - self.history_steps_1]
            joint_pos_at_2 = [0.0] * 12
            joint_vel_at_2 = [0.0] * 12
        else:
            joint_pos_at_1 = self.joint_pos_history[self.joint_history_pointer - self.history_steps_1]
            joint_vel_at_1 = self.joint_vel_history[self.joint_history_pointer - self.history_steps_1]
            joint_pos_at_2 = self.joint_pos_history[self.joint_history_pointer - self.history_steps_2]
            joint_vel_at_2 = self.joint_vel_history[self.joint_history_pointer - self.history_steps_2]

        base_velocity = self.parts['anymal'].speed()

        (x, y, z), (a, b, c, d), _, _, _, _, (vx, vy, vz), (vr, vp, vy) = self._p.getLinkState(
            dummy_base.bodies[dummy_base.bodyIndex], dummy_base.bodyPartIndex, computeLinkVelocity=1)
        body_angular_velocity = np.array([vr, vp, vz])

        # obs = np.concatenate(
        #     (gravity_b,[height] ,base_velocity, body_angular_velocity,joint_positions,joint_velocities,
        #      joint_pos_at_1,joint_pos_at_2,joint_vel_at_1,joint_vel_at_2,self.action_history[-2],self.cmd))

        if self.action_history_pointer < 1:
            obs = np.concatenate(
                (base_orientation_euler[1:],[height] ,base_velocity, body_angular_velocity,joint_positions,joint_velocities,
                 self.action_history[0]))
        else:
            obs = np.concatenate(
                (base_orientation_euler[1:], [height], base_velocity, body_angular_velocity, joint_positions,
                 joint_velocities,self.action_history[self.action_history_pointer-1]))

        joint_torque = [j.get_torch() for j in self.ordered_joints]
        self.joint_torque_history_pointer += 1
        self.joint_torque_history[self.joint_torque_history_pointer] = joint_torque

        foot_position = [self.parts['LF_ADAPTER'].get_position(), self.parts['RF_ADAPTER'].get_position(),
                         self.parts['LH_ADAPTER'].get_position(), self.parts['RH_ADAPTER'].get_position()]
        foot_position_z = [foot_position[0][2], foot_position[1][2], foot_position[2][2], foot_position[3][2]]

        foot_vel = [LA.norm(self.parts['LF_ADAPTER'].speed()[0:2]),
                    LA.norm(self.parts['RF_ADAPTER'].speed()[0:2]),
                    LA.norm(self.parts['LH_ADAPTER'].speed()[0:2]),
                    LA.norm(self.parts['RH_ADAPTER'].speed()[0:2])]
        if self.joint_torque_history_pointer < 1:
            anymal_state_dict = {'pos': base_position, 'vel': base_velocity, 'att': base_orientation_euler,
                                 'rate': body_angular_velocity, 'joint_pos': joint_positions,
                                 'joint_vel': joint_velocities,
                                 'joint_torque': joint_torque, 'foot_pos_z': foot_position_z, 'foot_vel': foot_vel,
                                 'vel_cmd': self.cmd,'gravity_vector':gravity_b,
                                 'delta_joint_torque': np.zeros((1,12))}
        else:

            anymal_state_dict = {'pos':base_position, 'vel':base_velocity,'att':base_orientation_euler,
                                 'rate': body_angular_velocity, 'joint_pos':joint_positions,'joint_vel':joint_velocities,
                                 'joint_torque':joint_torque,'foot_pos_z':foot_position_z,'foot_vel':foot_vel,'vel_cmd':self.cmd,'gravity_vector':gravity_b,
                                 'delta_joint_torque':self.joint_torque_history[self.joint_torque_history_pointer] - self.joint_torque_history[self.joint_torque_history_pointer-1]}

        base_contact_list = self.parts['anymal'].contact_list()

        return obs,anymal_state_dict
