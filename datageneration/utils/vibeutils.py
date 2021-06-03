import joblib
from glob import glob
import numpy as np
import os
import pickle as pkl

from utils.geometryutils import add_noise_poses, rotmat2rotvec, smooth_poses


def get_template_name(name, datasetname="ntu"):
    if datasetname == "ntu":
        template_name = "{}_rgb".format(name)
    elif datasetname == "uestc":
        template_name = name[:-4]
    else:
        template_name = name
    return template_name


def count_tracks(name, smpl_result_path, datasetname="ntu"):
    vibe_output = load_vibe_result(name, smpl_result_path, datasetname=datasetname)
    # This is not necessarily, 0, 1, 2... but weirdly 55, 56 etc.
    track_list = [*vibe_output]  # vibe_output.keys()
    num_tracks = len(track_list)
    return num_tracks, track_list


def load_vibe_result(name, vibe_path, datasetname="ntu"):
    template_name = get_template_name(name, datasetname)
    vibe_output_path = os.path.join(vibe_path, template_name, "vibe_output.pkl")
    if os.path.isfile(vibe_output_path):
        vibe_output = joblib.load(open(vibe_output_path, "rb"))
        return vibe_output
    else:
        print("Did not compute vibe for {}.".format(vibe_output_path))
        exit()


def load_smpl_body_data(
    name,
    smpl_result_path,
    track_id=0,
    with_trans=False,
    use_pose_smooth=True,
    use_z=False,
    datasetname="ntu",
    noise_factor=0.0,
    noise_level="video_level",
):
    """Load poses and trans"""
    vibe_output = load_vibe_result(name, smpl_result_path, datasetname=datasetname)
    vibe_output = vibe_output[track_id]
    num_frames = vibe_output["pose"].shape[0]
    # Poses in vibe are in axis-angle format as (num_frames, 72)
    vibe_output["poses"] = vibe_output["pose"]
    # Add trans key, init with zeros
    vibe_output["trans"] = np.zeros((num_frames, 3))
    for t in range(num_frames):
        if with_trans:
            vibe_output["trans"][t] = get_trans_from_vibe(vibe_output, t, use_z=use_z)
        # f * (3D + x) = 2D_x
        # f * (3D + y) = 2D_y
        # print(hmmr_output['cams'][t][0] * (hmmr_output['joints'][t, 2, 0] + hmmr_output['cams'][t][1]))
        # print(hmmr_output['kps'][t, 2, 0])
    # Subtract mean of all people's translations now in center_people
    # Subtract the mean translation to roughly center the person
    # mean_trans = hmmr_output['trans'].mean(axis=0)
    # hmmr_output['trans'] -= mean_trans
    vibe_output["trans"][:, 0] = smooth_signal(vibe_output["trans"][:, 0].squeeze())
    vibe_output["trans"][:, 1] = smooth_signal(vibe_output["trans"][:, 1].squeeze())
    if use_z:
        vibe_output["trans"][:, 2] = smooth_signal(vibe_output["trans"][:, 2].squeeze())
    # hmmr_output['trans'] += mean_trans

    # print(hmmr_output['trans'])
    # Temporal smoothing
    if use_pose_smooth:
        print("Temporal smoothing of poses.")
        vibe_output["poses"] = smooth_poses(vibe_output["poses"])

    if noise_factor != 0:
        print(
            "Adding noise to poses with a factor {}, level: {}".format(
                noise_factor, noise_level
            )
        )
        vibe_output["poses"] = add_noise_poses(
            vibe_output["poses"], noise_factor=noise_factor, level=noise_level
        )
    return vibe_output


def center_people(trans_data):
    mean_trans = np.concatenate(trans_data).mean(axis=0)
    for person_no in range(len(trans_data)):
        trans_data[person_no] -= mean_trans
    return trans_data


def smooth_signal(X):
    # Define the Gaussian kernel
    bin5mask = np.array([0.0625, 0.2500, 0.3750, 0.2500, 0.0625])
    bin5mask = np.array([0.15, 0.2, 0.3, 0.2, 0.15])
    bin9mask = np.convolve(bin5mask, bin5mask)
    bin17mask = np.convolve(bin9mask, bin9mask)
    kernel = bin17mask
    k = len(kernel)
    khalf = int((k - 1) / 2)
    temp = np.convolve(X, kernel, "valid")
    return np.concatenate((X[:khalf], temp, X[-khalf:]))


def get_z(cam_s, cam_pos, verts, img_size, flength):
    """
    Solves for the depth offset of the model to approx. orth with persp camera.
    From: Angjoo Kanazawa
    """
    # Translate the model itself: Solve the best z that maps to orth_proj points
    vert_orth_target = (cam_s * (verts[:, :2] + cam_pos) + 1) * 0.5 * img_size
    height3d = np.linalg.norm(
        np.max(verts[:, :2], axis=0) - np.min(verts[:, :2], axis=0)
    )
    height2d = np.linalg.norm(
        np.max(vert_orth_target, axis=0) - np.min(vert_orth_target, axis=0)
    )
    tz = np.array(flength * (height3d / height2d))
    return tz


def get_trans_from_vibe(vibe_output, t, use_z=True):
    # Convert crop cam to orig cam
    # No need! Because `convert_crop_cam_to_orig_img` from demoutils of vibe
    # does this already for us :)
    # Its format is: [sx, sy, tx, ty]
    cam_orig = vibe_output["orig_cam"][t]
    # cam_orig = get_orig_cam(hmmr_output['cams'][t], hmmr_output['bbox'][t]['start_pt'],
    #                         hmmr_output['bbox'][t]['scale'], hmmr_output['bbox'][t]['im_shape'])
    x = cam_orig[2]
    y = cam_orig[3]
    if use_z:
        z = get_z(
            cam_s=cam_orig[0],  # TODO: There are two scales instead of 1.
            cam_pos=cam_orig[2:4],
            verts=vibe_output["verts"][t],
            img_size=480,
            flength=500,
        )
        # z = 500 / (0.5 * 480 * cam_orig[0])
    else:
        z = 0
    return x, y, z


def get_orig_cam(cam, start_pt, scale, proc_img_shape):
    """
    Converts the cropped image cam to original image cam (squared).

    Args:
       start_pt, scale, proc_img_shape are parameters used to preprocess the
       image.

        start_pt=image_og_params['start_pt'], e.g. [419, 358]
        scale=image_og_params['scale'], e.g. 1.917
        proc_img_shape=image_og_params['im_shape'], e.g. [224, 224]
    """
    undo_scale = 1.0 / np.array(scale)
    img_size = 480
    # This is camera in crop image coord.
    cam_crop = np.hstack(
        [proc_img_shape[0] * cam[0] * 0.5, cam[1:] + (2.0 / cam[0]) * 0.5]
    )
    # This is camera in orig image coord
    cam_orig = np.hstack(
        [
            cam_crop[0] * undo_scale,
            cam_crop[1:] + (start_pt - proc_img_shape[0]) / cam_crop[0],
        ]
    )
    # This is the camera in normalized orig_image coord
    new_cam = np.hstack(
        [
            cam_orig[0] * (2.0 / img_size),
            cam_orig[1:] - (1 / ((2.0 / img_size) * cam_orig[0])),
        ]
    )
    new_cam = new_cam.astype(np.float32)
    return new_cam
