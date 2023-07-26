import torch
from torch import tensor
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from viz_robot import tm, RobotViz, FrameViz
import time
from robot_utils import torque_off
from robot import fkin_space


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

if __name__ == '__main__':

    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_right', init_node=True)

    torque_off(master_bot_right)
    from vpython import arrow, vector, canvas, cylinder, rate, sphere
    robot = RobotViz(joint_home_list, s_list)
    end_effector_viz = FrameViz(thickness=0.02)

    end_effector_origin = torch.tensor([0, 0, 0, 1.])
    end_effector_tip = torch.tensor([0.05, 0.0, 0., 1.])
    vecs = torch.stack((end_effector_origin, end_effector_tip), dim=1)
    end_effector = arrow(thickness=0.01, color=vector(0.1, 0.1, 1.))
    # end_effector = sphere(radius=0.1)

    def _v(tensor):
        return vector(tensor[0], tensor[1], tensor[2])

    while True:
        theta_list = torch.tensor(master_bot_right.dxl.joint_states.position[:6])

        robot.update(theta_list, fps=60)
        E = fkin_space(M, s_list, theta_list)

        end_pos = E.matmul(vecs)
        end_effector.pos = _v(end_pos[:, 0])
        end_effector.axis = _v(end_pos[:, 1] - end_pos[:, 0])

        # t += 0.05
        time.sleep(.1)