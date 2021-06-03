import numpy as np
import os
import joblib
import pickle as pkl
import scipy.io as sio
import sys
from tqdm import tqdm

proj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.append(proj_path)
import datasets.ntu


def load_vibe_result(name, vibe_path):
    template_name = "{}_rgb".format(name)
    vibe_output_path = os.path.join(vibe_path, template_name, "vibe_output.pkl")
    if os.path.isfile(vibe_output_path):
        vibe_output = joblib.load(open(vibe_output_path, "rb"))
        return vibe_output
    else:
        # print('Did not compute hmmr for {}.'.format(hmmr_output_path))
        return None


def main(setname="train", save_pose=False, save_joints=True, save_T=False):
    set_mapping = {"train": "train", "test": "valid"}
    # Ran once with test, once with train, then merged with merge script.
    # Data loader
    ntu_data = datasets.ntu.NTU(
        matfile="../../../data/ntu/info/ntu_data.mat",
        img_folder="../../../data/ntu",
        load_res=1,
        setname=setname,
        jointsIx=np.arange(24),
        pose_rep="xyz",
        views_list_str="0",
    )

    N = len(ntu_data.videos)  # 56576
    all_pose_smpl = []
    all_joints_smpl = []
    nframes_vibe = np.zeros(N)
    for ind in tqdm(range(N)):
        name = ntu_data._get_video_file(ind)
        name = ntu_data.videos[ind]
        vibe_output = load_vibe_result(
            name, "/home/gvarol/datasets/ntu/vibe/{}".format(setname)
        )
        if vibe_output is None:
            pose_smpl = []
            joints = []
            T_vibe = 0
        else:
            track_list = [*vibe_output]
            if len(track_list) == 0:
                pose_smpl = []
                joints = []
                T_vibe = 0
            else:
                # Take the first track
                vibe_output = vibe_output[track_list[0]]
                # keys: pred_cam, orig_cam, verts, pose, betas, joints3d, joints2d, bboxes, frame_ids
                T_vibe = vibe_output["verts"].shape[0]
                # verts_smpl = vibe_output['verts']  # (nframes, 6890, 3)
                joints = vibe_output["joints3d"]  # (nframes, 49, 3)
                pose_smpl = vibe_output["pose"]  # (nframes, 72)
                # shape_smpl = vibe_output["betas"]  # (nframes, 10)

        if save_pose:
            all_pose_smpl.append(pose_smpl)
        if save_joints:
            all_joints_smpl.append(joints)
        if save_T:
            nframes_vibe[ind] = T_vibe

    if save_pose:
        sio.savemat(
            f"/home/gvarol/datasets/ntu/pose_vibe_smpl_cache_{set_mapping[setname]}.mat",
            {"pose_vibe_smpl": all_pose_smpl},
        )
    if save_joints:
        sio.savemat(
            f"/home/gvarol/datasets/ntu/joints_vibe_smpl_cache_{set_mapping[setname]}.mat",
            {"joints_vibe_smpl": all_joints_smpl},
        )
    if save_T:
        sio.savemat(
            f"/home/gvarol/datasets/ntu/info/T_vibe_{set_mapping[setname]}.mat",
            {"T_vibe": nframes_vibe},
        )


if __name__ == "__main__":
    main(setname="train")
    main(setname="test")
