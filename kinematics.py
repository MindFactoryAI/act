import torch
from torch.nn.functional import mse_loss


def cross_matrix(omega):
    """
    Returns the skew symmetric matrix given a vector omega
    :param omega: 3D axis of rotation
    :return: skew symmetric matrix for axis
    given x, y, z returns
    [0, -z, y]
    [z, 0, -x]
    [-y, x, 0]
    """

    device = omega.device
    cross = torch.zeros(9, device=device).scatter(0, torch.tensor([5, 2, 1], device=device), omega).reshape(3, 3)
    cross = cross + cross.T
    return cross * torch.tensor([
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0]
    ], dtype=omega.dtype, device=device)


def matrix_exp_rotation(axis, theta):
    """
    rotation matrix around axis by angle theta
    :param axis: vector that points along the axis
    :param theta: N, angles of rotation around the axis
    :return: rotation matrix
    """
    N, device = theta.size(0), theta.device
    axis = torch.nan_to_num(axis / axis.norm())
    cross = cross_matrix(axis)
    axis, cross = axis.expand(N, 3), cross.expand(N, 3, 3)
    theta = theta.reshape(theta.size(0), 1, 1)
    rotation = torch.eye(3, device=device).expand(N, 3, 3) + torch.sin(theta) * cross + (1 - torch.cos(theta)) * cross.matmul(cross)
    return rotation


def matrix_exp_screw(screw, thetadot):
    """
    given a screw and batch of thetadots, compute the matrix exponential for each
    modern robotics eqn 3.88
    :param screw: 6, normalized screw axis such that either || angular velocity || == 1 or || velocity || == 1
    :param thetadot: N thetas
    :return: N translation matrices
    """
    N, device = thetadot.size(0), thetadot.device
    screw = screw.expand(N, 6).to(device)
    T = torch.eye(4, 4, device=device).repeat(N, 1, 1)
    v = screw[:, 3:6]
    R = matrix_exp_rotation(screw[0, 0:3], thetadot)
    w = cross_matrix(screw[0, 0:3]).expand(N, 3, 3)
    thetadot = thetadot.reshape(N, 1, 1)
    G = thetadot * torch.eye(3, 3, device=device).expand(N, 3, 3) + (1.0 - torch.cos(thetadot)) * w + (thetadot - torch.sin(thetadot)) * torch.matmul(w, w)
    v = G.matmul(v.unsqueeze(-1)).squeeze()
    T[:, 0:3, 0:3] = R
    T[:, 0:3, 3] = v
    return T


def fkin_space(M, s_list, theta_list):
    """
    Computes the position end effector given end_effector home position, list of screws in the space (inertial) frame
    and joint angles
    :param s_list: 6, J screws in the space frame
    :param theta_list: N, J joint angles (pose of the robot, for N robots)
    :return: (N, 4, 4) transformation matrix for the pose of the end effector in the body frame
    """

    _, J = s_list.shape
    N, device = theta_list.size(0), theta_list.device

    A = torch.eye(4, device=device).expand(N, 4, 4)
    for i in range(J):
        A = A.matmul(matrix_exp_screw(s_list[:, i], theta_list[:, i]))

    return A.matmul(M)


widow_x_250_6DOF_M = torch.tensor([
    [1., 0., 0., 0.458325],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.36065],
    [0., 0., 0., 1.],
])

widow_x_250_6DOF_s_list = torch.tensor([
    [0., 0., 1., 0., 0., 0.],
    [0., 1., 0., -0.11065, 0., 0.],
    [0., 1., 0., -0.36065, 0., 0.04975],
    [1., 0., 0., 0., 0.36065, 0.],
    [0., 1., 0., -0.36065, 0., 0.29975],
    [1., 0., 0., 0., 0.36065, 0.],
]).T


viper_x_300_6DOF_M = torch.tensor([
    [1., 0., 0., 0.536494],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.42705],
    [0., 0., 0., 1.],
])

viper_x_300_6DOF_s_list = torch.tensor([
    [0., 0., 1., 0.,       0.,      0.],
    [0., 1., 0., -0.12705, 0.,      0.],
    [0., 1., 0., -0.42705, 0.,      0.04975],
    [1., 0., 0., 0.,       0.42705, 0.],
    [0., 1., 0., -0.42705, 0.,      0.29975],
    [1., 0., 0., 0.,       0.42705, 0.],
]).T


def fk_loss_funky(actions_target, actions_hat):
    device = actions_target.device

    # get the arm joints and targets
    left_target, left_hat = actions_target[..., :7], actions_hat[..., :7]
    right_target, right_hat = actions_target[..., 7:], actions_hat[..., 7:]
    left_arm_target, left_arm_hat = left_target[..., :6], left_hat[..., :6]
    right_arm_target, right_arm_hat = right_target[..., :6], right_hat[..., :6]

    def compute_end_effector(left_arm, right_arm, M, s_list):
        # pack the joint list so we can send it through fkin in a single batch
        joint_list = torch.stack([left_arm, right_arm])
        joint_list_shape = joint_list.shape[:-1]
        joint_list = joint_list.flatten(start_dim=0, end_dim=-2)

        # get the frame of the end effector, and then compute the end effector pose
        end_effector_transform = fkin_space(M, s_list, joint_list)

        # now unpack
        end_effector_transform = end_effector_transform.unflatten(dim=0, sizes=joint_list_shape)
        left_arm, right_arm = tuple(end_effector_transform.unbind())
        return left_arm, right_arm

    left_arm_target, right_arm_target = compute_end_effector(
        left_arm_target, right_arm_target,  M=widow_x_250_6DOF_M.to(device), s_list=widow_x_250_6DOF_s_list.to(device))
    left_arm_hat, right_arm_hat = compute_end_effector(
        left_arm_hat, right_arm_hat, M=viper_x_300_6DOF_M.to(device), s_list=viper_x_300_6DOF_s_list.to(device))

    targets = torch.stack([left_arm_target, right_arm_target], dim=-1)
    hats = torch.stack([left_arm_hat, right_arm_hat], dim=-1)

    # compute the l1 norm between the end effector poses
    arm_loss = torch.abs(targets.detach() - hats)
    left_loss, right_loss = arm_loss[..., 0], arm_loss[..., 1]
    arm_loss = (left_loss + right_loss).mean(-1).mean(-1)

    # now do the grippers
    left_gripper_target, left_gripper_hat = left_target[..., -1], left_hat[..., -1]
    right_gripper_target, right_gripper_hat = right_target[..., -1], right_hat[..., -1]
    gripper_targets = torch.stack([left_gripper_target, right_gripper_target], dim=-1)
    gripper_hats = torch.stack([left_gripper_hat, right_gripper_hat], dim=-1)
    gripper_loss = torch.abs(gripper_targets.detach() - gripper_hats)
    gripper_left_loss, gripper_right_loss = gripper_loss[..., 0], gripper_loss[..., 1]
    gripper_loss = gripper_left_loss + gripper_right_loss

    return arm_loss + gripper_loss


def fk_loss(actions_target, actions_hat, M, s_list):
    device = actions_target.device

    # get the arm joints and targets
    left_target, left_hat = actions_target[..., :7], actions_hat[..., :7]
    right_target, right_hat = actions_target[..., 7:], actions_hat[..., 7:]
    left_arm_target, left_arm_hat = left_target[..., :6], left_hat[..., :6]
    right_arm_target, right_arm_hat = right_target[..., :6], right_hat[..., :6]

    # pack the joint list so we can send it through fkin in a single batch
    joint_list = torch.stack([left_arm_target, right_arm_target, left_arm_hat, right_arm_hat])
    joint_list_shape = joint_list.shape[:-1]
    joint_list = joint_list.flatten(start_dim=0, end_dim=-2)

    # get the frame of the end effector, and then compute the end effector pose
    end_effector_transform = fkin_space(M, s_list, joint_list)

    # now unpack and repack
    end_effector_transform = end_effector_transform.unflatten(dim=0, sizes=joint_list_shape)
    left_arm_target, right_arm_target, left_arm_hat, right_arm_hat = tuple(end_effector_transform.unbind())
    targets = torch.stack([left_arm_target, right_arm_target], dim=-1)
    hats = torch.stack([left_arm_hat, right_arm_hat], dim=-1)

    # compute the l1 norm between the end effector poses
    arm_loss = torch.abs(targets.detach() - hats)
    left_loss, right_loss = arm_loss[..., 0], arm_loss[..., 1]
    arm_loss = (left_loss + right_loss).mean(-1).mean(-1)

    # now do the grippers
    left_gripper_target, left_gripper_hat = left_target[..., -1], left_hat[..., -1]
    right_gripper_target, right_gripper_hat = right_target[..., -1], right_hat[..., -1]
    gripper_targets = torch.stack([left_gripper_target, right_gripper_target], dim=-1)
    gripper_hats = torch.stack([left_gripper_hat, right_gripper_hat], dim=-1)
    gripper_loss = torch.abs(gripper_targets.detach() - gripper_hats)
    gripper_left_loss, gripper_right_loss = gripper_loss[..., 0], gripper_loss[..., 1]
    gripper_loss = gripper_left_loss + gripper_right_loss

    return arm_loss + gripper_loss


def rotation_mat(translation_matrix):
    return translation_matrix[:, 0:3, 0:3]


def translation_vec(translation_matrix):
    return translation_matrix[:, 0:3, 3:]


def screw_distance(s1, s2, alpha=1):

    # Poses
    p1, R1 = translation_vec(s1), rotation_mat(s1)
    p2, R2 = translation_vec(s2), rotation_mat(s2)

    # Screw components
    R1_t = torch.transpose(R1, 1, 2)
    s = torch.matmul(R1_t, p2 - p1).squeeze(-1)

    # cosine distance
    tr = torch.matmul(R1_t, R2).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    eps = torch.finfo(tr.dtype).eps
    tr = (tr - 1) / 2
    tr = tr.clamp(-1. + eps, 1. - eps)
    theta = torch.acos(tr)

    return (s * s).sum(-1) + alpha * theta ** 2

