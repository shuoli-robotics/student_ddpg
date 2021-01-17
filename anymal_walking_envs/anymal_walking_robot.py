from ksluck_sim_private.pybullet_ksluck.anymal_base import AnymalRobot
import numpy as np

class AnymalWalkRobot(AnymalRobot):
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
                # print(j.power_coef)
                joint_pos = self.convert2pos(a[n],j.upperLimit,j.lowerLimit)
                j.set_pd_torque(joint_pos, 40.0, 1.0)
        else:
            raise ValueError('No known control mode')

    def convert2pos(self,action,joint_max_pos,joint_min_pos):
        dis2min = action + 1
        prop = dis2min / 2.0
        return prop * (joint_max_pos-joint_min_pos) + joint_min_pos
