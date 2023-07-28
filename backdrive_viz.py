import torch
from torch import tensor
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import time
from robot_utils import torque_off
from kinematics import fkin_space, matrix_exp_screw
from vpython import arrow, vector, canvas, cylinder, rate, distant_light, color


x = tensor([1., 0., 0.])
y = tensor([0., 1., 0.])
z = tensor([0., 0., 1.])


def _r(x, y, z):
    return torch.stack((x, y, z), dim=0)


M = torch.tensor([
    [1., 0., 0., 0.458325],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.36065],
    [0., 0., 0., 1.],
])

Z1, Z2 = 0.11065, 0.36065


def tm(x=0.0, y=0.0, z=0.0, R=None):
    """ returns a translation matrix """
    t = torch.tensor([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ], dtype=torch.float)
    if R is not None:
        t[0:3, 0:3] = R
    return t


joint_home_list = [
    tm(),
    tm(x=0.00000, z=0.11065, R=_r(x, z, y)),
    tm(x=0.04975, z=0.36065, R=_r(x, z, y)),
    tm(x=0.22155, z=0.36065, R=_r(z, y, x)),
    tm(x=0.30000, z=0.36065, R=_r(x, z, y)),
    tm(x=0.36500, z=0.36065, R=_r(z, y, x)),
]

s_list = torch.tensor([
    [0., 0., 1., 0., 0., 0.],
    [0., 1., 0., -0.11065, 0., 0.],
    [0., 1., 0., -0.36065, 0., 0.04975],
    [1., 0., 0., 0., 0.36065, 0.],
    [0., 1., 0., -0.36065, 0., 0.29975],
    [1., 0., 0., 0., 0.36065, 0.],
]).T


frame = torch.tensor([
    [0, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
], dtype=torch.float).T


def origin(frame):
    return _v(frame[0:3, 0])


def z_axis(frame):
    return _v(frame[0:3, 3])


class RobotViz:
    def __init__(self, joint_home_list, s_list, num_robots):

        self.joints = joint_home_list
        self.n_joints = len(joint_home_list)
        self.screws = s_list

        self.scene = canvas(width=1200, height=1200)
        self.scene.camera.pos = vector(0, -2.4, -0.2)
        self.scene.camera.axis = vector(0, 2.4, 0.27)
        self.scene.camera.up = vector(0, -1, 0)
        self.scene = distant_light(direction=vector(-2, 0, 0), color=color.white)
        self.scene = distant_light(direction=vector(2, 0, 0), color=color.white)

        self.joints_viz = []
        self.links_viz = []

        for _ in range(num_robots):
            self.joints_viz.append([cylinder(axis=vector(0, 0, 0.0), radius=0.04, color=vector(0.1, 0.3, 0.4), visible=False) for _ in range(self.n_joints)])
            self.links_viz.append([None] + [cylinder(radius=0.03, color=vector(0.6, 0.6, 0.6), visible=False) for _ in range(self.n_joints-1)])

    def update(self, theta, fps=24):

        ts = [matrix_exp_screw(self.screws[:, i], theta[:, i]) for i in range(self.n_joints)]
        self.draw_single_robot(ts, 0, tm(x=.5, R=_r(-x, y, z)))
        self.draw_single_robot(ts, 1, tm(x=-.5))
        rate(fps)

    def draw_single_robot(self, ts, i, base_transform):

        tb = base_transform
        prev_joint_pos = vector(0, 0, 0)
        end_frame = None

        for tf, j, jv, lv in zip(ts, self.joints, self.joints_viz[i], self.links_viz[i]):
            tb = tb.matmul(tf[i])
            joint_frame = tb.matmul(j).matmul(frame)

            jv.visible = True
            jv.pos = origin(joint_frame)
            jv.axis = (z_axis(joint_frame) - origin(joint_frame)) * 0.05
            if lv:
                lv.visible = True
                lv.pos = prev_joint_pos
                axis = jv.pos - prev_joint_pos
                if axis.mag == 0:
                    lv.visible = False
                else:
                    lv.axis = axis

            prev_joint_pos = origin(joint_frame)
            end_frame = joint_frame
        return end_frame


if __name__ == '__main__':

    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)

    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_right', init_node=False)

    torque_off(master_bot_right)
    robot = RobotViz(joint_home_list, s_list, num_robots=2)

    base_frames = torch.stack([tm(x=.5, R=_r(-x, y, z)), tm(x=-.5)])
    end_effector_origin = torch.tensor([0, 0, 0, 1.])
    end_effector_tip = torch.tensor([0.05, 0.0, 0., 1.])

    vecs = torch.stack((end_effector_origin, end_effector_tip), dim=1)
    vecs = vecs.expand(2, 4, 2)
    end_effectors = [arrow(thickness=0.01, color=vector(1., 0.5, 1.)), arrow(thickness=0.01, color=vector(0.2,  1., .2))]

    def _v(tensor):
        return vector(tensor[0], tensor[1], tensor[2])

    while True:
        theta_list = torch.tensor([master_bot_right.dxl.joint_states.position[:6], master_bot_left.dxl.joint_states.position[:6]])

        robot.update(theta_list, fps=60)

        E = fkin_space(M, s_list, theta_list)
        end_pos = base_frames.matmul(E.matmul(vecs))

        for i, vec in enumerate(vecs):
            end_effectors[i].pos = _v(end_pos[i, :, 0])
            end_effectors[i].axis = _v(end_pos[i, :, 1] - end_pos[i, :, 0])

        # t += 0.05
        time.sleep(.1)