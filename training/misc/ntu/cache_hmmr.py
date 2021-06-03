import numpy as np
import os
import pickle as pkl
import scipy.io as sio
import sys
from tqdm import tqdm

proj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.append(proj_path)
import datasets.ntu


def load_hmmr_result(name, hmmr_path):
    template_name = "{}_rgb".format(name)
    hmmr_output_dir = os.path.join(hmmr_path, template_name)
    # hmmr_output_paths = glob(os.path.join(hmmr_output_dir, 'hmmr_output*.pkl'))
    # num_tracks = len(hmmr_output_paths)
    # if num_tracks > 1:
    #     return None
    # First track
    hmmr_output_path = os.path.join(hmmr_output_dir, "hmmr_output.pkl")
    if os.path.isfile(hmmr_output_path):
        hmmr_output = pkl.load(open(hmmr_output_path, "rb"))
    else:
        # print('Did not compute hmmr for {}.'.format(hmmr_output_path))
        return None
    return hmmr_output


def main(setname="train", save_pose=True, save_joints=True, save_T=True):
    # Ran once with test, once with train, then merged with merge script.
    set_mapping = {"train": "train", "test": "valid"}
    # J_regressor
    smpl = pkl.load(
        open(
            "../../../utils/smpl/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
            "rb",
        ),
        encoding="latin1",
    )
    smpl_J_regressor = smpl["J_regressor"]  # (24, 6890)

    # Data loader
    ntu_data = datasets.ntu.NTU(
        matfile="../../../data/ntu/info/ntu_data.mat",
        img_folder="../../../data/ntu",
        load_res=1,
        setname=setname,
        views_list_str="0",
    )

    N = len(ntu_data.videos)  # 56576
    all_pose_smpl = []
    all_pose_rot_smpl = []
    all_joints_smpl = []
    all_joints_coco = []
    nframes_hmmr = np.zeros(N)
    for ind in tqdm(range(N)):
        name = ntu_data.videos[ind]
        hmmr_output = load_hmmr_result(
            name, "/home/gvarol/datasets/ntu/hmmr/{}".format(setname)
        )
        if hmmr_output is None:
            pose_smpl = []
            pose_rot_smpl = []
            joints_smpl = []
            joints_coco = []
            T_hmmr = 0
        else:
            T_hmmr = hmmr_output["verts"].shape[0]
            verts_smpl = hmmr_output["verts"]  # (nframes, 6890, 3)
            joints_coco = hmmr_output["joints"]  # (nframes, 25, 3)
            # (nframes, 85) [theta(72) beta(10) cam(3)]
            pose_smpl = hmmr_output["omegas"]
            # (nframes, 24, 3, 3) rotation matrices
            pose_rot_smpl = hmmr_output["poses"]
            joints_smpl = np.zeros((T_hmmr, 24, 3))
            for t in range(T_hmmr):
                joints_smpl[t] = smpl_J_regressor.dot(verts_smpl[t])  # (nframes, 24, 3)
            # Needs to multiply dimensions (1, 2) by -1 before training

        if save_pose:
            all_pose_smpl.append(pose_smpl)
            all_pose_rot_smpl.append(pose_rot_smpl)
        if save_T:
            nframes_hmmr[ind] = T_hmmr
        if save_joints:
            all_joints_smpl.append(joints_smpl)
            all_joints_coco.append(joints_coco)

    if save_pose:
        sio.savemat(
            f"/home/gvarol/datasets/ntu/pose_hmmr_smpl_cache_{set_mapping[setname]}.mat",
            {"pose_hmmr_smpl": all_pose_smpl, "pose_rot_hmmr_smpl": all_pose_rot_smpl},
        )
    if save_joints:
        sio.savemat(
            f"/home/gvarol/datasets/ntu/joints3D_hmmr_smpl_cache_{set_mapping[setname]}.mat",
            {"joints3D_hmmr_smpl": all_joints_smpl},
        )
        sio.savemat(
            f"/home/gvarol/datasets/ntu/joints3D_hmmr_coco_cache_{set_mapping[setname]}.mat",
            {"joints3D_hmmr_coco": all_joints_coco},
        )
    if save_T:
        sio.savemat(
            f"/home/gvarol/datasets/ntu/info/T_hmmr_{set_mapping[setname]}.mat",
            {"T_hmmr": nframes_hmmr},
        )


if __name__ == "__main__":
    main(setname="train")
    main(setname="test")
