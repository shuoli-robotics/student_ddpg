from pybullet_envs.scene_stadium import SinglePlayerStadiumScene
from pybullet_envs.env_bases import MJCFBaseBulletEnv
import numpy as np
import pybullet
# from anymal_base import AnymalRobot


class AnymalBaseBulletEnv(MJCFBaseBulletEnv):

  def __init__(self, robot, render=True):
    # print("WalkerBase::__init__ start")
    self.camera_x = 0
    self.walk_target_x = 1e3  # kilometer away
    self.walk_target_y = 0
    self.stateId = -1
    MJCFBaseBulletEnv.__init__(self, robot, render)


  def create_single_player_scene(self, bullet_client):
    self.stadium_scene = SinglePlayerStadiumScene(bullet_client,
                                                  gravity=9.8,
                                                  timestep=0.0165 / 4,
                                                  frame_skip=4)
    return self.stadium_scene

  def reset(self):
    if (self.stateId >= 0):
      print("restoreState self.stateId:",self.stateId)
      self._p.restoreState(self.stateId)

    r = MJCFBaseBulletEnv.reset(self)
    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

    self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(
        self._p, self.stadium_scene.ground_plane_mjcf)
    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
    if (self.stateId < 0):
      self.stateId = self._p.saveState()
      print("saving state self.stateId:",self.stateId)

    self.camera_adjust()
    self.robot.robot_specific_reset(self._p)

    return r

  def _isDone(self):
    return False #self._alive < 0

  def move_robot(self, init_x, init_y, init_z):
      pass
      #   "Used by multiplayer stadium to move sideways, to another running lane."
      #   self.cpp_robot.query_position()
      #   pose = self.cpp_robot.root_part.pose()
      #   pose.move_xyz(
      #       init_x, init_y, init_z
      #   )  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
      #   self.cpp_robot.set_pose(pose)
      #
      # electricity_cost = -2.0  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
      # stall_torque_cost = -0.1  # cost for running electric current through a motor even at zero rotational speed, small
      # foot_collision_cost = -1.0  # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
      # foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
      # joints_at_limit_cost = -0.1  # discourage stuck joints

  def step(self, a):
    self.robot.apply_action(a)
    self.scene.global_step()

    state = self.robot.calc_state()  # also calculates self.joints_at_limit
    done = self._isDone()
    # if not np.isfinite(state).all():
    #   print("~INF~", state)
    #   done = True

    reward = 0

    return state, reward, bool(done), {}

  def camera_adjust(self):
    x, y, z = self.robot.body_xyz
    self.camera_x = 0.98 * self.camera_x + (1 - 0.98) * x
    self.camera.move_and_look_at(self.camera_x, y - 2.0, 1.4, x, y, 1.0)
