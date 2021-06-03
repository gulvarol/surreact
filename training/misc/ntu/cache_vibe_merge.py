import scipy.io as sio

# TRAIN
joints_train = sio.loadmat(
    "/home/gvarol/datasets/ntu/joints_vibe_smpl_cache_train.mat"
)["joints_vibe_smpl"][0]
T_train = sio.loadmat("/home/gvarol/datasets/ntu/info/T_vibe_train.mat")["T_vibe"][0]
pose_train = sio.loadmat("/home/gvarol/datasets/ntu/pose_vibe_smpl_cache_train.mat")[
    "pose_vibe_smpl"
][0]

# TEST
joints_valid = sio.loadmat(
    "/home/gvarol/datasets/ntu/joints_vibe_smpl_cache_valid.mat"
)["joints_vibe_smpl"][0]
T_valid = sio.loadmat("/home/gvarol/datasets/ntu/info/T_vibe_valid.mat")["T_vibe"][0]
pose_valid = sio.loadmat("/home/gvarol/datasets/ntu/pose_vibe_smpl_cache_valid.mat")[
    "pose_vibe_smpl"
][0]

N = len(joints_train)
for i in range(N):
    if len(joints_train[i]) == 0:
        joints_train[i] = joints_valid[i]
        T_train[i] = T_valid[i]
        pose_train[i] = pose_valid[i]


sio.savemat(
    "/home/gvarol/datasets/ntu/joints_vibe_smpl_cache.mat",
    {"joints_vibe_smpl": joints_train},
)
sio.savemat("/home/gvarol/datasets/ntu/info/T_vibe.mat", {"T_vibe": T_train})
sio.savemat(
    "/home/gvarol/datasets/ntu/pose_vibe_smpl_cache.mat", {"pose_vibe_smpl": pose_train}
)
