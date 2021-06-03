import scipy.io as sio

# POSE
pose_hmmr_smpl_train = sio.loadmat(
    "/home/gvarol/datasets/ntu/pose_hmmr_smpl_cache_train.mat"
)["pose_hmmr_smpl"][0]
pose_hmmr_smpl_valid = sio.loadmat(
    "/home/gvarol/datasets/ntu/pose_hmmr_smpl_cache_valid.mat"
)["pose_hmmr_smpl"][0]

pose_rot_hmmr_smpl_train = sio.loadmat(
    "/home/gvarol/datasets/ntu/pose_hmmr_smpl_cache_train.mat"
)["pose_rot_hmmr_smpl"][0]
pose_rot_hmmr_smpl_valid = sio.loadmat(
    "/home/gvarol/datasets/ntu/pose_hmmr_smpl_cache_valid.mat"
)["pose_rot_hmmr_smpl"][0]

# JOINTS
joints_hmmr_smpl_train = sio.loadmat(
    "/home/gvarol/datasets/ntu/joints3D_hmmr_smpl_cache_train.mat"
)["joints3D_hmmr_smpl"][0]
joints_hmmr_smpl_valid = sio.loadmat(
    "/home/gvarol/datasets/ntu/joints3D_hmmr_smpl_cache_valid.mat"
)["joints3D_hmmr_smpl"][0]

joints_hmmr_coco_train = sio.loadmat(
    "/home/gvarol/datasets/ntu/joints3D_hmmr_coco_cache_valid.mat"
)["joints3D_hmmr_coco"][0]
joints_hmmr_coco_valid = sio.loadmat(
    "/home/gvarol/datasets/ntu/joints3D_hmmr_coco_cache_valid.mat"
)["joints3D_hmmr_coco"][0]

# T
T_train = sio.loadmat("/home/gvarol/datasets/ntu/info/T_hmmr_train.mat")["T_hmmr"][0]
T_valid = sio.loadmat("/home/gvarol/datasets/ntu/info/T_hmmr_valid.mat")["T_hmmr"][0]

N = len(pose_hmmr_smpl_train)
for i in range(N):
    if len(pose_hmmr_smpl_train[i]) == 0 and len(pose_hmmr_smpl_valid[i]) != 0:
        pose_hmmr_smpl_train[i] = pose_hmmr_smpl_valid[i]
        pose_rot_hmmr_smpl_train[i] = pose_rot_hmmr_smpl_valid[i]
        joints_hmmr_smpl_train[i] = joints_hmmr_smpl_valid[i]
        joints_hmmr_coco_train[i] = joints_hmmr_coco_valid[i]
        T_train[i] = T_valid[i]

sio.savemat(
    "/home/gvarol/datasets/ntu/pose_hmmr_smpl_cache.mat",
    {
        "pose_hmmr_smpl": pose_hmmr_smpl_train,
        "pose_rot_hmmr_smpl": pose_rot_hmmr_smpl_train,
    },
)

sio.savemat(
    "/home/gvarol/datasets/ntu/joints3D_hmmr_smpl_cache.mat",
    {"joints3D_hmmr_smpl": joints_hmmr_smpl_train},
)

sio.savemat(
    "/home/gvarol/datasets/ntu/joints3D_hmmr_coco_cache.mat",
    {"joints3D_hmmr_coco": joints_hmmr_coco_train},
)

sio.savemat("/home/gvarol/datasets/ntu/info/T_hmmr.mat", {"T_hmmr": T_train})
