import os
import math
import numpy as np
import scipy.io as sio

import torch
from datasets.videodataset import VideoDataset
from utils.imutils import change_segmix, indices_to_onehot
from utils.misc import to_torch


class SURREACT(VideoDataset):
    def __init__(
        self,
        img_folder,
        matfile="surreact_data.mat",
        inp_res=256,
        setname="train",
        scale_factor=0.25,
        num_in_frames=1,
        pose_rep="vector",
        evaluate_video=False,
        jointsIx=[],
        num_crops=1,
        hflip=0,
        views_list_str="0",
        randframes=False,
        use_segm="",
        use_flow="",
        randbgvid=False,
    ):
        print("{} => Views list: {}".format(setname, views_list_str))
        self.img_folder = img_folder
        self.setname = setname  # train, val or test
        self.inp_res = inp_res
        self.scale_factor = scale_factor
        self.num_in_frames = num_in_frames
        self.pose_rep = pose_rep
        self.evaluate_video = evaluate_video
        self.hflip = hflip
        self.num_crops = num_crops
        self.randframes = randframes
        self.use_segm = use_segm  # '' | 'as_input' | 'mask_rgb'
        self.use_flow = use_flow  # '' | 'as_input
        self.randbgvid = randbgvid

        self.segmIx = np.array((2, 12, 9, 2, 13, 10, 2, 14, 11, 2, 14, 11, 2, 2, 2, 1, 6, 3, 7, 4, 8, 5, 8, 5))

        self.jointsIx = jointsIx

        self.img_width = 320
        self.img_height = 240

        # Load names, T, splits
        print("Loading {}".format(matfile))
        dict_data = sio.loadmat(
            os.path.join(img_folder, "info", matfile), squeeze_me=True
        )
        self.videos = [s.strip() for s in dict_data["names"]]
        self.num_frames = dict_data["T"]
        if "uestc/vibe" in img_folder:
            import pickle as pkl

            print("Workaround of loading safe num_frames")
            num_frames_pkl = os.path.join(img_folder, "info", "num_frames.pkl")
            self.num_frames = pkl.load(open(num_frames_pkl, "rb"))
        self.train = dict_data["splits"][0]
        self.valid = dict_data["splits"][1]

        N = len(self.videos)

        # Is it NTU or HRI40?
        self.realdataset = self.img_folder.split("/")[-2]
        if self.realdataset == "ntu":
            self.setups = np.zeros(N, dtype=int)
            self.cameras = np.zeros(N, dtype=int)
            self.subjects = np.zeros(N, dtype=int)
            self.repetitions = np.zeros(N, dtype=int)
            self.actions = np.zeros(N, dtype=int)
            self.views = np.zeros(N, dtype=int)
            for ind in range(N):
                (
                    self.setups[ind],
                    self.cameras[ind],
                    self.subjects[ind],
                    self.repetitions[ind],
                    self.actions[ind],
                    self.views[ind],
                ) = self._get_ntu_setup_camera_subject_repetition_action_view(
                    self.videos[ind]
                )
        elif self.realdataset == "uestc":
            self.actions = np.zeros(N, dtype=int)
            self.views = np.zeros(N, dtype=int)
            for ind in range(N):
                self.actions[ind], self.views[ind] = self._get_hri40_action_view(
                    self.videos[ind]
                )
        else:
            raise ValueError(
                f"Unrecognized {self.realdataset} as the real data source of this synthetic dataset?"
            )

        views_list = [int(x) for x in views_list_str.split("-")]

        num_views = len(views_list)

        self.train_tmp = np.array([]).astype(int)
        self.valid_tmp = np.array([]).astype(int)
        # For each view in the views_list
        for v in range(num_views):
            v1_ix = np.where(self.views == views_list[v])[0]
            train_v1_ix = np.isin(v1_ix, self.train)
            valid_v1_ix = np.isin(v1_ix, self.valid)
            self.train_tmp = np.append(self.train_tmp, v1_ix[train_v1_ix])
            self.valid_tmp = np.append(self.valid_tmp, v1_ix[valid_v1_ix])
        self.train = self.train_tmp
        self.valid = self.valid_tmp

        # Take a subset of 100 samples
        if self.setname == "val" and len(self.valid) > 100:
            self.valid = self.valid[:: int(len(self.valid) / 100)]

        if evaluate_video:
            self.valid, self.t_beg = self._slide_windows(self.valid, self.num_crops)

        # self.rgb_pad_method = 'zeros'
        self.rgb_pad_method = "copy_last"
        self._parse_what_to_load()
        self._set_action_classes()
        VideoDataset.__init__(self)

    def _set_datasetname(self):
        self.datasetname = "surreact"

    def _get_infofile(self, ind):
        return self._get_video_file(ind)[:-4] + "_info.mat"

    def _get_video_file(self, ind):
        if self.setname == "val":
            set_path = "test"
        else:
            set_path = self.setname
        return os.path.join(
            self.img_folder, set_path, self.videos[ind][:-13], self.videos[ind]
        )

    def _get_mat_file(self, ind, s):
        return self._get_video_file(ind)[:-4] + s + ".mat"

    def _get_ntu_setup_camera_subject_repetition_action_view(self, video_name):
        setup = int(video_name[1:4])
        camera = int(video_name[5:8])
        subject = int(video_name[9:12])
        repetition = int(video_name[13:16])
        action = int(video_name[17:20]) - 1
        if len(video_name) == 33:
            view = int(video_name[22:25])
        elif len(video_name) == 54:  # this is should cover both cases
            view = int(video_name[-11:-8])
        else:
            print("Which videoname format? {}".format(video_name))
        # rep
        return setup, camera, subject, repetition, action, view

    def _get_hri40_action_view(self, video_name):
        # repetition = int(video_name[-6:-4])
        action = video_name.split("_")[0][1:]
        view = int(video_name[-11:-8])
        return action, view

    def _get_action(self, ind):
        return self.actions[ind]

    def _load_joints2D(self, ind, t):
        infofile = self._get_infofile(ind)
        # Load joints2D variable for all frames (num_people, 2, 24, 100)
        joints2D = sio.loadmat(infofile, variable_names=["joints2D"])["joints2D"]  
        # Surreact version v06, v07 ... are with multi-person
        if joints2D.ndim == 4 or (joints2D.ndim == 3 and joints2D.shape[1] == 2):
            # Take the first person for now
            joints2D = joints2D[0]  # (2, 24, 100)
        if joints2D.ndim == 2:  # single frame
            joints2D = joints2D.reshape(joints2D.shape[0], joints2D.shape[1], 1)
        joints2D = joints2D[:, :, t].transpose()  # (24, 2)
        left_right_ix = np.array((1, 3, 2, 4, 6, 5, 7, 9, 8, 10, 12, 11, 13, 15, 14, 16, 18, 17, 20, 19, 22, 21, 24, 23)) - 1
        joints2D = joints2D[left_right_ix, :]
        return joints2D[self.jointsIx, :]  # (16, 2)

    def _load_joints3D(self, ind, frame_ix):
        infofile = self._get_infofile(ind)
        # Load joints3D variable for all frames (num_people, 3, 24, 100)
        joints3D = sio.loadmat(infofile, variable_names=["joints3D"])["joints3D"]
        # Surreact version v06, v07 ... are with multi-person
        if joints3D.ndim == 4 or (joints3D.ndim == 3 and joints3D.shape[1] == 3):
            joints3D = joints3D[0]  # (3, 24, 100)
        if joints3D.ndim == 2:  # single frame
            joints3D = joints3D.reshape(joints3D.shape[0], joints3D.shape[1], 1)
        joints3D = np.transpose(joints3D[:, :, frame_ix], (1, 0, 2))  # (24, 3, nframes)
        # joints3D[:, 1] = -joints3D[:, 1]
        joints3D[:, 2] = -joints3D[:, 2]
        joints3D = joints3D[:, (0, 2, 1)]
        joints3D = joints3D - joints3D[0]  # subtract pelvis
        # left_right_ix = np.array(
        #     (1, 3, 2, 4, 6, 5, 7, 9, 8, 10, 12, 11, 13, 15, 14, 16, 18, 17, 20, 19, 22, 21, 24, 23)) - 1
        # joints3D = joints3D[left_right_ix, :]
        return to_torch(joints3D[self.jointsIx].flatten()).float()  # (16, 3, nframes)

    def _get_extrinsics(self, zrot_euler, cam_height, cam_dist):
        from mathutils import Matrix, Euler

        rot_z = math.radians(zrot_euler)
        rot_x = math.atan((cam_height - 1) / cam_dist)
        # Rotate -90 degrees around x axis to have the person face cam
        cam_rot = Matrix(((1, 0, 0), (0, 0, 1), (0, -1, 0))).to_4x4()
        # Rotation by zrot_euler around z-axis
        cam_rot_z = Euler((0, rot_z, 0), "XYZ").to_matrix().to_4x4()
        cam_rot_x = Euler((rot_x, 0, 0), "XYZ").to_matrix().to_4x4()

        # Rotate around the object by rot_z with a fixed radius = cam_dist
        cam_trans = Matrix.Translation(
            [cam_dist * math.sin(rot_z), cam_dist * math.cos(rot_z), cam_height]
        )

        cam_ext = cam_trans * cam_rot * cam_rot_z * cam_rot_x
        return cam_ext

    def _load_segm(self, ind, frame_ix, is_onehot=False):
        nframes = len(frame_ix)
        varnames = ["segm_%d" % (tt + 1) for tt in frame_ix]
        dict_segm = sio.loadmat(
            self._get_mat_file(ind, "_segm"), variable_names=varnames
        )
        segm = torch.zeros(nframes, self.img_height, self.img_width)
        for f, fix in enumerate(frame_ix):
            segm[f] = to_torch(dict_segm[varnames[f]]).int()
            segm[f] = change_segmix(segm[f], self.segmIx)
            if is_onehot:
                segm[f] = indices_to_onehot(segm[f])
        return segm

    def _load_flow(self, ind, frame_ix):
        nframes = len(frame_ix)
        varnames = ["gtflow_%d" % (tt + 1) for tt in frame_ix]
        dict_flow = sio.loadmat(
            self._get_mat_file(ind, "_gtflow"),
            variable_names=varnames,
            verify_compressed_data_integrity=False,
        )
        flow = torch.zeros(2, nframes, self.img_height, self.img_width)
        for f, fix in enumerate(frame_ix):
            flow[:, f] = (
                to_torch(dict_flow[varnames[f]].transpose(2, 0, 1)).contiguous().float()
            )
        # Invert the x-channel
        flow[0] *= -1
        return flow

    def _load_pose(self, ind, frame_ix, pose_rep="vector"):
        if pose_rep == "xyz":
            # return self._load_joints3D(ind, t, nframes=1)
            return {"joints3D": self._load_joints3D(ind, frame_ix)}
        else:
            raise ValueError(f"Pose representation {pose_rep} undefined.")
