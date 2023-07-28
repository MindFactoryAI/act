import torch
from kinematics import matrix_exp_screw, rotation_mat, translation_vec, screw_distance
from math import sin, cos
from matplotlib import pyplot as plt

poly = torch.tensor([
    [-1., +1, +1, -1, -1],
    [+1., +1, -1, -1, +1],
    [+0., +0, -0, -0, +0],
    [1., 1, 1, 1, 1.]
])


def test_matrix_exp_translate():
    """ translate 1 unit/sec in y axis"""
    s = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float)
    t = matrix_exp_screw(s, torch.tensor([1., 0.]))
    assert t[0, 1, 3].item() == 1.0

    ax = plt.subplot()
    ax.plot()


def lerp_cube_by_screw(screw, start, end, steps):
    transforms = matrix_exp_screw(screw, torch.linspace(start, end, steps))
    plt.ion()
    fig, ax = plt.subplots()
    poly_screen = transforms[0].matmul(poly)
    cube = ax.plot(poly_screen[0], poly_screen[1])
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))

    for t in transforms:
        poly_screen = t.matmul(poly)
        cube[0].set_data(poly_screen[0], poly_screen[1])
        plt.pause(0.2)


def test_viz_matrix_exp_translate():
    """ up """
    lerp_cube_by_screw(torch.tensor([0., 0, 0, 0, 1, 0]), 0., 1., 20)

    """ down """
    lerp_cube_by_screw(torch.tensor([0., 0, 0, 0, -1, 0]), 0., 1., 20)

    """ left """
    lerp_cube_by_screw(torch.tensor([0., 0, 0, -1, 0, 0]), 0., 1., 20)

    """ right """
    lerp_cube_by_screw(torch.tensor([0., 0, 0, 1, 0, 0]), 0., 1., 20)

    """ rotate counter """
    lerp_cube_by_screw(torch.tensor([0., 0, 1, 0, 0, 0]), 0., 1., 20)

    """ rotate clockwise """
    lerp_cube_by_screw(torch.tensor([0., 0, -1, 0, 0, 0]), 0., 1., 20)


def test_matrix_exp_rotate():
    """ rotate at 1 rad/sec around z axis"""
    s = torch.tensor([0.0, 0, 1.0, 0, 0, 0], dtype=torch.float)
    t = matrix_exp_screw(s, torch.tensor([1.0]))
    frame = torch.tensor([
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ], dtype=torch.float).T
    frame = t.matmul(frame)
    assert frame[0, 1, 1].item() - sin(1.0) < 0.0001
    assert frame[0, 2, 2].item() - cos(1.0) < 0.0001


def test_screw_distance():
    t1 = torch.tensor([
        [0., 0, 1, 1],
        [1., 0, 0, 2],
        [0, 1, 0, 3],
        [0., 0., 0., 1.]
    ]).unsqueeze(0)

    t2 = torch.tensor([
        [0., 1, 0, 4],
        [1., 0, 0, 5],
        [0., 0, 1, 6],
        [0., 0., 0, 1.]
    ]).unsqueeze(0)

    p1 = torch.tensor([1, 2, 3]).unsqueeze(0).unsqueeze(-1)
    R1 = torch.tensor([[0, 0, 1],
                       [1, 0, 0],
                       [0, 1, 0]]).unsqueeze(0)

    p2 = torch.tensor([4, 5, 6]).unsqueeze(0).unsqueeze(-1)
    R2 = torch.tensor([[0, 1, 0],
                       [1, 0, 0],
                       [0, 0, 1]]).unsqueeze(0)

    assert (p1 == translation_vec(t1)).all()
    assert (p2 == translation_vec(t2)).all()
    assert (R1 == rotation_mat(t1)).all()
    assert (R2 == rotation_mat(t2)).all()

    d = screw_distance(t1, t2)

    t3 = torch.tensor([
        [0, 1., 0, 0],
        [1., 0, 0, 0],
        [0., 0, 1, 0],
        [0., 0., 0, 1.]
    ], requires_grad=True).unsqueeze(0)

    d = screw_distance(torch.eye(4).unsqueeze(0), t3)
    d.mean().backward()

    print(d, t3.grad)