import numpy as np

import utils.rotations


def kintree_table():
    # Kinematic tree: (parent, joint)
    return np.array(
        (
            (-1, 0),
            (0, 1),
            (20, 2),
            (2, 3),
            (20, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (20, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (0, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (0, 16),
            (16, 17),
            (17, 18),
            (18, 19),
            (1, 20),
            (7, 21),
            (7, 22),
            (11, 23),
            (11, 24),
        )
    ).transpose()


def tree_traverse():
    # From root to leaves, layers of the tree
    return np.array(
        (
            (1, 12, 16),
            (20, 13, 17),
            (2, 4, 8, 14, 18),
            (3, 5, 9, 15, 19),
            (6, 10),
            (7, 11),
            (21, 22, 23, 24),
        )
    )


def endpoints():
    # Rotation always [1 0 0], so we don't predict them
    return np.array([3, 15, 19, 21, 22, 23, 24])


def free_joints():
    # Rotations to predict
    return (
        np.array((1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21)) - 1
    )


def bone_lengths():
    # Sample bone lengths
    # dict_mat = sio.loadmat('bone_lengths.mat')
    # return dict_mat['bone_lengths']
    return np.array(
        (
            0.0,
            0.26990068,
            0.06685794,
            0.13500227,
            0.14567094,
            0.18454897,
            0.20564714,
            0.02626529,
            0.13262259,
            0.20777225,
            0.25388875,
            0.06208725,
            0.05583357,
            0.33694179,
            0.32458302,
            0.07992818,
            0.07610215,
            0.32879907,
            0.31463922,
            0.08735712,
            0.20109801,
            0.02871096,
            0.04079429,
            0.04659919,
            0.03643913,
        )
    )


def angles2xyz(angles_rotmat, blen=None):
    if blen is None:
        blen = bone_lengths()
    kintree = kintree_table()
    treetrav = tree_traverse()
    # Find xyz given angles and bone_lengths
    xyz_trans = np.zeros((25, 3))
    for i in range(treetrav.shape[0]):
        for j in treetrav[i]:
            parent = kintree[0][j]
            xyz_trans[j] = xyz_trans[parent] + np.dot(
                angles_rotmat[j], np.array((0, blen[j], 0))
            )
    return xyz_trans


def xyz2abs(xyz):
    kintree = kintree_table()
    treetrav = tree_traverse()
    # Find absolute angles given xyz
    angles_rotmat = np.zeros((25, 3, 3))
    for i in range(treetrav.shape[0]):
        for j in treetrav[i]:
            parent = kintree[0][j]
            blen = np.sum((xyz[j] - xyz[parent]) ** 2) ** 0.5  # bone length
            v1 = np.array((0, blen, 0))
            v2 = xyz[j] - xyz[parent]
            angle_rotvec = utils.rotations.axisangle(v1, v2)
            angles_rotmat[j] = utils.rotations.rotvec2rotmat(angle_rotvec)
    return angles_rotmat


def xyz2rel(xyz):
    kintree = kintree_table()
    treetrav = tree_traverse()
    # Find relative angles given xyz
    angles_rotmat = np.zeros((25, 3, 3))
    angles_rotmat_rel = np.zeros((25, 3, 3))
    angles_rotmat_rel[0] = np.eye(3)
    angles_rotmat[0] = np.eye(3)
    for i in range(treetrav.shape[0]):
        for j in treetrav[i]:
            parent = kintree[0][j]
            blen = np.sum((xyz[j] - xyz[parent]) ** 2) ** 0.5  # bone length
            v1 = np.array((0, blen, 0))
            v2 = xyz[j] - xyz[parent]
            angle_rotvec = utils.rotations.axisangle(v1, v2)
            angles_rotmat[j] = utils.rotations.rotvec2rotmat(angle_rotvec)
            angles_rotmat_rel[j] = np.dot(
                angles_rotmat[parent].transpose(), angles_rotmat[j]
            )
    return angles_rotmat_rel


def abs2rel(angles_abs):
    kintree = kintree_table()
    treetrav = tree_traverse()
    # Find relative angles given absolute angles
    # In rotation matrix representation
    angles_rel = np.zeros((25, 3, 3))
    angles_rel[0] = angles_abs[0]
    for i in range(treetrav.shape[0]):
        for j in treetrav[i]:
            parent = kintree[0][j]
            angles_rel[j] = np.dot(angles_abs[parent].transpose(), angles_abs[j])
    return angles_rel


def rel2abs(angles_rel, root_given=False):
    kintree = kintree_table()
    treetrav = tree_traverse()
    # Find absolute angles given relative angles
    # In rotation matrix representation
    angles_abs = np.zeros((25, 3, 3))
    if root_given:
        angles_abs[0] = angles_rel[0]
    else:
        angles_abs[0] = np.eye(3)
    for i in range(treetrav.shape[0]):
        for j in treetrav[i]:
            parent = kintree[0][j]
            angles_abs[j] = np.dot(angles_abs[parent], angles_rel[j])
    return angles_abs


def output2xyz(angles_rotmat, pose_rep="rotmat"):
    freejoints = free_joints()
    angles_rotmat_valid = np.zeros((25, 3, 3))
    # If relative, convert to absolute for plotting
    if pose_rep == "rotmat":
        angles_rel = np.zeros((25, 3, 3))
        angles_rel[freejoints] = angles_rotmat
        angles_abs = rel2abs(angles_rel)
        angles_rotmat = angles_abs[freejoints]

    for j, jnt in enumerate(freejoints):
        angles_rotmat_valid[jnt] = utils.rotations.makeRotationMatrix(angles_rotmat[j])

    xyz = angles2xyz(angles_rotmat_valid)
    return xyz
