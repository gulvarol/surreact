from glob import glob
import numpy as np
import os
import pickle as pkl

from utils.geometryutils import add_noise_poses, rotmat2rotvec, smooth_poses


def count_tracks(name, smpl_result_path, datasetname="ntu"):
    if datasetname == "ntu":
        template_name = "{}_rgb".format(name)
    elif datasetname == "uestc":
        template_name = name[:-4]
    else:
        template_name = name
    hmmr_output_dir = os.path.join(smpl_result_path, template_name)
    hmmr_output_paths = glob(os.path.join(hmmr_output_dir, "hmmr_output*.pkl"))
    num_tracks = len(hmmr_output_paths)

    track_list = []
    for h in sorted(hmmr_output_paths):
        if os.path.basename(h) == "hmmr_output.pkl":
            track_list.append(0)
        else:
            # hmmr_output_x.pkl we retrieve x
            track_list.append(int(os.path.basename(h).split("_")[2][:-4]))
    return num_tracks, track_list


def load_hmmr_result(name, hmmr_path, track_id=0, datasetname="ntu"):
    if datasetname == "ntu":
        template_name = "{}_rgb".format(name)
    elif datasetname == "uestc":
        template_name = name[:-4]
    else:
        template_name = name

    if track_id == 0:
        hmmr_output_path = os.path.join(hmmr_path, template_name, "hmmr_output.pkl")
        hmmr_bbox_path = os.path.join(hmmr_path, template_name, "hmmr_bbox.pkl")
    else:
        assert track_id > 0
        hmmr_output_path = os.path.join(
            hmmr_path, template_name, "hmmr_output_{}.pkl".format(track_id)
        )
        hmmr_bbox_path = os.path.join(
            hmmr_path, template_name, "hmmr_bbox_{}.pkl".format(track_id)
        )
    if os.path.isfile(hmmr_output_path) and os.path.isfile(hmmr_bbox_path):
        hmmr_output = pkl.load(open(hmmr_output_path, "rb"))
        hmmr_output["bbox"] = pkl.load(open(hmmr_bbox_path, "rb"))
        return hmmr_output
    else:
        print(
            "Did not compute hmmr for {} or {}.".format(
                hmmr_output_path, hmmr_bbox_path
            )
        )
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
    hmmr_output = load_hmmr_result(
        name, smpl_result_path, track_id=track_id, datasetname=datasetname
    )
    num_frames = hmmr_output["poses"].shape[0]
    # Add trans key, init with zeros
    hmmr_output["trans"] = np.zeros((num_frames, 3))
    # Poses in hmmr are in rotmat representation
    hmmr_output["poses_rotmat"] = hmmr_output["poses"]
    # Add axis-angle poses
    hmmr_output["poses"] = np.zeros((num_frames, 24, 3))
    for t in range(num_frames):
        # Convert rotmat to axis-angle for each joint
        for j in range(24):
            hmmr_output["poses"][t][j] = rotmat2rotvec(
                hmmr_output["poses_rotmat"][t, j]
            )

        if with_trans:
            hmmr_output["trans"][t] = get_trans_from_hmmr(hmmr_output, t, use_z=use_z)
        # f * (3D + x) = 2D_x
        # f * (3D + y) = 2D_y
        # print(hmmr_output['cams'][t][0] * (hmmr_output['joints'][t, 2, 0] + hmmr_output['cams'][t][1]))
        # print(hmmr_output['kps'][t, 2, 0])
    # Subtract mean of all people's translations now in center_people
    # Subtract the mean translation to roughly center the person
    # mean_trans = hmmr_output['trans'].mean(axis=0)
    # hmmr_output['trans'] -= mean_trans
    hmmr_output["trans"][:, 0] = smooth_signal(hmmr_output["trans"][:, 0].squeeze())
    hmmr_output["trans"][:, 1] = smooth_signal(hmmr_output["trans"][:, 1].squeeze())
    if use_z:
        hmmr_output["trans"][:, 2] = smooth_signal(hmmr_output["trans"][:, 2].squeeze())
    # hmmr_output['trans'] += mean_trans

    hmmr_output["poses"] = hmmr_output["poses"].reshape(num_frames, 72)
    # print(hmmr_output['trans'])
    # Temporal smoothing
    if use_pose_smooth:
        print("Temporal smoothing of poses.")
        hmmr_output["poses"] = smooth_poses(hmmr_output["poses"])

    if noise_factor != 0:
        print(
            "Adding noise to poses with a factor {}, level: {}".format(
                noise_factor, noise_level
            )
        )
        hmmr_output["poses"] = add_noise_poses(
            hmmr_output["poses"], noise_factor=noise_factor, level=noise_level
        )
    return hmmr_output


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


def get_trans_from_hmmr(hmmr_output, t, use_z=True):
    # Convert crop cam to orig cam
    cam_orig = get_orig_cam(
        hmmr_output["cams"][t],
        hmmr_output["bbox"][t]["start_pt"],
        hmmr_output["bbox"][t]["scale"],
        hmmr_output["bbox"][t]["im_shape"],
    )
    x = cam_orig[1]
    y = cam_orig[2]
    if use_z:
        z = get_z(
            cam_s=cam_orig[0],
            cam_pos=cam_orig[1:],
            verts=hmmr_output["verts"][t],
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
