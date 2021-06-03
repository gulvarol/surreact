import math
import numpy as np

_FLOAT_EPS = np.finfo(np.float).eps


# computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec / theta).reshape(3, 1) if theta > 0.0 else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    return cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat


# transformation between pose and blendshapes
def rodrigues2bshapes(pose):
    rod_rots = np.asarray(pose).reshape(24, 3)
    mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]
    bshapes = np.concatenate(
        [(mat_rot - np.eye(3)).ravel() for mat_rot in mat_rots[1:]]
    )
    return mat_rots, bshapes


def normalize(v):
    v_mag = np.linalg.norm(v)
    if v_mag == 0:
        v = np.zeros(3)
        v[0] = 1
    else:
        v = v / v_mag
    return v


def rotmat2rotvec(mat, unit_thresh=1e-5):
    """Return axis, angle and point from (3, 3) matrix `mat`
    Parameters
    ----------
    mat : array-like shape (3, 3)
        Rotation matrix
    unit_thresh : float, optional
        Tolerable difference from 1 when testing for unit eigenvalues to
        confirm `mat` is a rotation matrix.
    Returns
    -------
    axis : array shape (3,)
       vector giving axis of rotation
    angle : scalar
       angle of rotation in radians.
    Examples
    --------
    >>> direc = np.random.random(3) - 0.5
    >>> angle = (np.random.random() - 0.5) * (2*math.pi)
    >>> R0 = axangle2mat(direc, angle)
    >>> direc, angle = mat2axangle(R0)
    >>> R1 = axangle2mat(direc, angle)
    >>> np.allclose(R0, R1)
    True
    Notes
    -----
    http://en.wikipedia.org/wiki/Rotation_matrix#Axis_of_a_rotation
    Code from https://github.com/matthew-brett/transforms3d
    """
    M = np.asarray(mat, dtype=np.float)
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    L, W = np.linalg.eig(M.T)
    i = np.where(np.abs(L - 1.0) < unit_thresh)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = np.real(W[:, i[-1]]).squeeze()
    # rotation angle depending on direction
    cosa = (np.trace(M) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (M[1, 0] + (cosa - 1.0) * direction[0] * direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (M[0, 2] + (cosa - 1.0) * direction[0] * direction[2]) / direction[1]
    else:
        sina = (M[2, 1] + (cosa - 1.0) * direction[1] * direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    # return direction, angle
    # print(np.linalg.norm(direction))  # 1
    # assert(np.linalg.norm(direction) == 1.0)
    return direction * angle


def axangle2quat(vector, theta, is_normalized=False):
    """Quaternion for rotation of angle `theta` around `vector`
    Parameters
    ----------
    vector : 3 element sequence
       vector specifying axis for rotation.
    theta : scalar
       angle of rotation in radians.
    is_normalized : bool, optional
       True if vector is already normalized (has norm of 1).  Default
       False.
    Returns
    -------
    quat : 4 element sequence of symbols
       quaternion giving specified rotation
    Examples
    --------
    >>> q = axangle2quat([1, 0, 0], np.pi)
    >>> np.allclose(q, [0, 1, 0,  0])
    True
    Notes
    -----
    Formula from http://mathworld.wolfram.com/EulerParameters.html
    Code from https://github.com/matthew-brett/transforms3d
    """
    vector = np.array(vector)
    if not is_normalized:
        # Cannot divide in-place because input vector may be integer type,
        # whereas output will be float type; this may raise an error in versions
        # of numpy > 1.6.1
        vector = vector / math.sqrt(np.dot(vector, vector))
    t2 = theta / 2.0
    st2 = math.sin(t2)
    return np.concatenate(([math.cos(t2)], vector * st2))


def quat2axangle(quat, identity_thresh=None):
    """Convert quaternion to rotation of angle around axis
    Parameters
    ----------
    quat : 4 element sequence
       w, x, y, z forming quaternion.
    identity_thresh : None or scalar, optional
       Threshold below which the norm of the vector part of the quaternion (x,
       y, z) is deemed to be 0, leading to the identity rotation.  None (the
       default) leads to a threshold estimated based on the precision of the
       input.
    Returns
    -------
    theta : scalar
       angle of rotation.
    vector : array shape (3,)
       axis around which rotation occurs.
    Examples
    --------
    >>> vec, theta = quat2axangle([0, 1, 0, 0])
    >>> vec
    array([1., 0., 0.])
    >>> np.allclose(theta, np.pi)
    True
    If this is an identity rotation, we return a zero angle and an arbitrary
    vector:
    >>> quat2axangle([1, 0, 0, 0])
    (array([1., 0., 0.]), 0.0)
    If any of the quaternion values are not finite, we return a NaN in the
    angle, and an arbitrary vector:
    >>> quat2axangle([1, np.inf, 0, 0])
    (array([1., 0., 0.]), nan)
    Notes
    -----
    A quaternion for which x, y, z are all equal to 0, is an identity rotation.
    In this case we return a 0 angle and an arbitrary vector, here [1, 0, 0].
    The algorithm allows for quaternions that have not been normalized.
    Code from https://github.com/matthew-brett/transforms3d
    """
    w, x, y, z = quat
    Nq = w * w + x * x + y * y + z * z
    if not np.isfinite(Nq):
        return np.array([1.0, 0, 0]), float("nan")
    if identity_thresh is None:
        try:
            identity_thresh = np.finfo(Nq.type).eps * 3
        except (AttributeError, ValueError):  # Not a numpy type or not float
            identity_thresh = _FLOAT_EPS * 3
    if Nq < _FLOAT_EPS ** 2:  # Results unreliable after normalization
        return np.array([1.0, 0, 0]), 0.0
    if Nq != 1:  # Normalize if not normalized
        s = math.sqrt(Nq)
        w, x, y, z = w / s, x / s, y / s, z / s
    len2 = x * x + y * y + z * z
    if len2 < identity_thresh ** 2:
        # if vec is nearly 0,0,0, this is an identity rotation
        return np.array([1.0, 0, 0]), 0.0
    # Make sure w is not slightly above 1 or below -1
    theta = 2 * math.acos(max(min(w, 1), -1))
    return np.array([x, y, z]) / math.sqrt(len2), theta


def rotmat2rotvec2(R):
    u = np.zeros(3)
    x = 0.5 * (R[0, 0] + R[1, 1] + R[2, 2] - 1)  # 1.0000000484288
    x = max(x, -1)
    x = min(x, 1)
    theta = math.acos(x)  # Tr(R) = 1 + 2 cos(theta)

    if theta < 1e-4:  # avoid division by zero!
        # print('theta ~= 0 %f' % theta)
        return u
    elif abs(theta - math.pi) < 1e-4:
        # print('theta ~= pi %f' % theta)
        if R[0][0] >= R[2][2]:
            if R[1][1] >= R[2][2]:
                u[0] = R[0][0] + 1
                u[1] = R[1][0]
                u[2] = R[2][0]
            else:
                u[0] = R[0][1]
                u[1] = R[1][1] + 1
                u[2] = R[2][1]
        else:
            u[0] = R[0][2]
            u[1] = R[1][2]
            u[2] = R[2][2] + 1

        u = normalize(u)
    else:
        d = 1 / (2 * math.sin(theta))  # ||u|| = 2sin(theta)
        u[0] = d * (R[2, 1] - R[1, 2])
        u[1] = d * (R[0, 2] - R[2, 0])
        u[2] = d * (R[1, 0] - R[0, 1])
    return u * theta


def axangle2quat_batch(poses):
    """
    poses: (nframes, 24, 3)
    """
    nframes, njoints, ndims = poses.shape
    poses_quat = np.zeros((nframes, njoints, 4))
    for t in range(nframes):
        for j in range(njoints):
            angle = np.linalg.norm(poses[t, j])
            axis = poses[t, j] / angle
            poses_quat[t, j] = axangle2quat(axis, angle, is_normalized=True)
    return poses_quat


def quat2axangle_batch(poses):
    """
    poses: (nframes, 24, 4)
    """
    nframes, njoints, ndims = poses.shape
    poses_axangle = np.zeros((nframes, njoints, 3))
    for t in range(nframes):
        for j in range(njoints):
            axis, angle = quat2axangle(poses[t, j])
            poses_axangle[t, j] = axis * angle
    return poses_axangle


def smooth_poses(poses):
    """
    Args: poses (nframes, 72)
    """
    # Define the Gaussian kernel
    bin3mask = np.array([2 / 3, 1, 2 / 3])
    bin3mask = bin3mask / bin3mask.sum()
    bin5mask = np.array([0.0625, 0.2500, 0.3750, 0.2500, 0.0625])
    bin9mask = np.convolve(bin5mask, bin5mask)
    bin17mask = np.convolve(bin9mask, bin9mask)
    kernel = bin3mask
    k = len(kernel)
    khalf = int((k - 1) / 2)
    assert poses.ndim == 2
    nframes, ndims = poses.shape
    njoints = 24
    # Convert to quaternions
    poses_quat = axangle2quat_batch(poses.reshape(nframes, njoints, 3)).reshape(
        nframes, njoints * 4
    )
    ndims_quat = poses_quat.shape[1]
    poses_quat_sm = poses_quat.copy()
    # For all joints * 4, except the first one (global rotation)
    # Maybe the comment remained from early versions? It includes global rotation now.
    for d in range(0, ndims_quat):
        # Smooth over time on the quaternion
        temp = np.convolve(poses_quat[:, d], kernel, "valid")
        poses_quat_sm[:, d] = np.concatenate(
            (poses_quat[:khalf, d], temp, poses_quat[-khalf:, d])
        )

    poses_quat_sm = poses_quat_sm.reshape(nframes, njoints, 4)
    for t in range(0, nframes):
        for j in range(njoints):
            quat_unnorm = poses_quat_sm[t, j]
            m = np.linalg.norm(quat_unnorm)
            poses_quat_sm[t, j] = poses_quat_sm[t, j] / m

    # Convert the smoothed quaternions to axis-angle
    poses_sm = quat2axangle_batch(poses_quat_sm)
    return poses_sm.reshape(nframes, njoints * 3)


def add_noise_poses(poses, level="video_level", noise_factor=0.05):
    """
    Args: poses (nframes, 72)
    """
    # Define the Gaussian kernel
    assert poses.ndim == 2
    nframes, ndims = poses.shape
    njoints = 24
    # Convert to quaternions
    poses_quat = axangle2quat_batch(poses.reshape(nframes, njoints, 3)).reshape(
        nframes, njoints * 4
    )
    ndims_quat = poses_quat.shape[1]
    poses_quat_sm = poses_quat.copy()
    # For all joints * 4
    for d in range(0, ndims_quat):
        # noise range is [-0.05, 0.05] for noise_factor=0.05
        if level == "video_level":
            # Add the same noise to all frames
            noise = noise_factor * (2 * np.random.rand() - 1)
        elif level == "independent_frames":
            # Add different noise to each frames
            noise = noise_factor * (2 * np.random.rand(nframes) - 1)
        elif level == "interpolate_frames":
            # Add different noise to a select number of frames, interpolate in between
            frame_skips = 25
            key_frames = np.arange(0, nframes + 1, frame_skips)
            if key_frames[-1] != nframes:
                key_frames = np.append(key_frames, nframes)
            noise_key = noise_factor * (2 * np.random.rand(len(key_frames)))
            noise = np.array([])
            for i, _ in enumerate(key_frames[:-1]):
                noise = np.append(
                    noise,
                    np.linspace(noise_key[i], noise_key[i + 1], np.diff(key_frames)[i]),
                )
        else:
            raise ValueError("Unrecognized level {}".format(level))
        poses_quat_sm[:, d] = poses_quat[:, d] + noise

    poses_quat_sm = poses_quat_sm.reshape(nframes, njoints, 4)
    for t in range(0, nframes):
        for j in range(njoints):
            quat_unnorm = poses_quat_sm[t, j]
            m = np.linalg.norm(quat_unnorm)
            poses_quat_sm[t, j] = poses_quat_sm[t, j] / m

    # Convert the smoothed quaternions to axis-angle
    poses_sm = quat2axangle_batch(poses_quat_sm)
    return poses_sm.reshape(nframes, njoints * 3)
