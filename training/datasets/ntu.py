import os
import numpy as np
import scipy.io as sio

import torch
from datasets.videodataset import VideoDataset

import utils.kinect.kinectutils as kinect
import utils.rotations
from utils.misc import to_torch


class NTU(VideoDataset):
    def __init__(
        self,
        matfile,
        img_folder,
        inp_res=256,
        setname="train",
        scale_factor=0.1,
        num_in_frames=1,
        pose_rep="vector",
        evaluate_video=False,
        load_res=1,  # was set to 0.25 for paper experiments
        protocol="",
        jointsIx=[],
        num_crops=1,
        hflip=0,
        views_list_str="0",
        randframes=False,
        input_type="rgb",
        joints_source="hmmr",
    ):
        print("{} => Views list: {}".format(setname, views_list_str))
        self.img_folder = img_folder
        self.setname = setname  # train, val or test
        self.inp_res = inp_res
        self.scale_factor = scale_factor
        self.num_in_frames = num_in_frames
        self.pose_rep = pose_rep
        self.evaluate_video = evaluate_video
        self.load_res = load_res
        self.protocol = protocol
        self.hflip = hflip
        self.num_crops = num_crops
        self.randframes = randframes
        self.input_type = input_type
        self.joints_source = joints_source  # kinect, hmmr, vibe

        self.jointsIx = jointsIx

        self.img_width = int(1920 * self.load_res)
        self.img_height = int(1080 * self.load_res)

        # Load pre-computed #frames data
        dict_nframes = sio.loadmat(matfile, variable_names=["T", "T_skeleton"])
        num_frames_video = dict_nframes["T"].squeeze()
        num_frames_skeleton = dict_nframes["T_skeleton"].squeeze()
        ntu_num_frames = np.minimum(num_frames_video, num_frames_skeleton)

        self.tr_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
        self.test_subjects = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40]

        # Remove 302 + 2 videos without skeleton annotations
        with open(
            os.path.join(self.img_folder, "info/samples_with_missing_skeletons.txt"),
            "r",
        ) as f:
            missing_skeletons = f.read().splitlines()
        missing_skeletons.append("S013C003P019R002A027")
        missing_skeletons.append("S012C003P015R002A060")

        # Load names of 56880 videos
        ntu_names = sio.loadmat(
            os.path.join(img_folder, "info", "ntu_names.mat"), squeeze_me=True
        )
        ntu_names = ntu_names["ntu_names"]

        # Remaining 56578 - 2 videos after rm missing skeletons
        self.videos = []
        not_missing_ix = []
        for i, name in enumerate(ntu_names):
            if name not in missing_skeletons:
                self.videos.append(name)
                not_missing_ix.append(i)

        # Load cached skeleton data
        self._parse_what_to_load()
        if self.load_info["joints2D"]:
            print("Loading cached Kinect 2D joints")
            self.joints2D_kinect = sio.loadmat(
                os.path.join(img_folder, "joints2D_kinect_cache.mat")
            )["joints2D_kinect"]
            self.joints2D_kinect = self.joints2D_kinect[:, not_missing_ix]
        if self.load_info["joints3D"]:
            print("Loading cached Kinect 3D joints")
            self.joints3D_kinect = sio.loadmat(
                os.path.join(img_folder, "joints3D_kinect_cache.mat")
            )["joints3D_kinect"]
            self.joints3D_kinect = self.joints3D_kinect[:, not_missing_ix]

            if self.joints_source == "hmmr":
                print("Loading cached HMMR SMPL 3D joints")
                self.joints3D_smpl = sio.loadmat(
                    os.path.join(img_folder, "joints3D_hmmr_smpl_cache.mat")
                )["joints3D_hmmr_smpl"]

                print("Loading cached HMMR SMPL pose parameters")
                self.pose_smpl = sio.loadmat(
                    os.path.join(img_folder, "pose_hmmr_smpl_cache.mat")
                )["pose_hmmr_smpl"]

            if self.joints_source == "vibe":
                print("Loading cached VIBE SMPL pose parameter")
                self.pose_smpl = sio.loadmat(
                    os.path.join(img_folder, "pose_vibe_smpl_cache.mat")
                )["pose_vibe_smpl"]

                print("Loading cached VIBE SMPL/SPIN (49d) joints")
                self.joints3D_smpl = sio.loadmat(
                    os.path.join(img_folder, "joints_vibe_smpl_cache.mat")
                )["joints_vibe_smpl"]

        if True:  # self.load_info['pose']:
            print("Loading cached Kinect angles")
            self.pose_kinect = sio.loadmat(
                os.path.join(img_folder, "pose_kinect_cache.mat")
            )["pose_kinect"]
            self.pose_kinect = self.pose_kinect[:, not_missing_ix]

        # Load pre-computed #frames data
        self.num_frames = ntu_num_frames[not_missing_ix]

        if self.load_info["joints3D"]:
            if self.joints_source == "hmmr":
                num_frames_joints = sio.loadmat(
                    os.path.join(img_folder, "info", "T_hmmr.mat"),
                    variable_names=["T_hmmr"],
                )["T_hmmr"].squeeze()
            elif self.joints_source == "vibe":
                num_frames_joints = sio.loadmat(
                    os.path.join(img_folder, "info", "T_vibe.mat"),
                    variable_names=["T_vibe"],
                )["T_vibe"].squeeze()
            else:
                print("Check if this problem exists with Kinect joints")
            if self.joints_source == "hmmr" or self.joints_source == "vibe":
                self.num_frames = np.minimum(self.num_frames, num_frames_joints).astype(
                    int
                )
            hmmr_extracted_ix = np.where(self.num_frames != 0)[0].tolist()
        else:
            hmmr_extracted_ix = []

        N = len(self.videos)
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
            ) = self._get_setup_camera_subject_repetition_action_view(self.videos[ind])

        # Camera split. 0: 18932, 1: 18847, 2: 18797
        self.camera_split = []
        for i in [1, 2, 3]:
            self.camera_split.append(np.where(self.cameras == i)[0])

        # View split. 0: 18889, 45: 18932, 90: 18755
        degrees = {0: 0, 45: 1, 90: 2}
        self.view_split = []
        for i in [0, 45, 90]:
            self.view_split.append(np.where(self.views == i)[0])

        # Subject split. 0: 40089, 1: 16487
        self.subject_split = []
        self.subject_split.append(np.where(np.isin(self.subjects, self.tr_subjects))[0])
        self.subject_split.append(
            np.where(np.isin(self.subjects, self.test_subjects))[0]
        )

        self._cache_sync_views()

        # First branch
        self.train = []
        self.valid = []
        views_list = [degrees[int(x)] for x in views_list_str.split("-")]
        for v in range(len(views_list)):
            if protocol == "CV":
                # Make sure you train with cameras 1&2, i.e. views 0-90
                # assert(views_list_str == '0')
                # assert(test_views == '90')
                self.train += list(self.sync_views[views_list[v]])
                # Standard test split of CV
                self.valid = list(self.camera_split[0])
            elif protocol == "CS":
                # Use this for testing
                self.train += list(
                    self.sync_views[views_list[v]][
                        np.where(
                            np.isin(
                                self.sync_views[views_list[v]], self.subject_split[0]
                            )
                        )[0]
                    ]
                )
                # Standard test split of CS
                self.valid = self.subject_split[1]
            else:
                # 13225/5447 for each view train/test
                self.train += list(
                    self.sync_views[views_list[v]][
                        np.where(
                            np.isin(
                                self.sync_views[views_list[v]], self.subject_split[0]
                            )
                        )[0]
                    ]
                )
                self.valid += list(
                    self.sync_views[views_list[v]][
                        np.where(
                            np.isin(
                                self.sync_views[views_list[v]], self.subject_split[1]
                            )
                        )[0]
                    ]
                )

        if self.load_info["joints3D"]:
            self.train = list(set(self.train) & set(hmmr_extracted_ix))
            self.valid = list(set(self.valid) & set(hmmr_extracted_ix))

        # Take a subset of 100 samples
        if self.setname == "val" and len(self.valid) > 100:
            self.valid = self.valid[:: int(len(self.valid) / 100)]
        # Redundant for now
        elif self.setname == "test":
            self.valid = self.valid

        if evaluate_video:
            self.valid, self.t_beg = self._slide_windows(self.valid, self.num_crops)

        self.rgb_pad_method = "zeros"
        self._set_action_classes()
        VideoDataset.__init__(self)

    def _cache_sync_views(self):
        cache_file = os.path.join(self.img_folder, "info", "sync_views.mat")
        if os.path.isfile(cache_file):
            dict_mat = sio.loadmat(cache_file)
            self.sync_views = dict_mat["sync_views"]
        else:
            print("Creating sync cache file at {}".format(cache_file))
            v1 = []
            v2 = []
            v3 = []
            cnt_short = 0
            cnt_nonsync = 0
            for k, v in enumerate(self.view_split[0]):
                v1_name = self._get_video_name_by_view(self.setups[v], 0, self.subjects[v], self.repetitions[v], self.actions[v])
                v2_name = self._get_video_name_by_view(self.setups[v], 45, self.subjects[v], self.repetitions[v], self.actions[v])
                v3_name = self._get_video_name_by_view(self.setups[v], 90, self.subjects[v], self.repetitions[v], self.actions[v])
                if (
                    v1_name in self.videos
                    and v2_name in self.videos
                    and v3_name in self.videos
                ):
                    v1_ix = self.videos.index(v1_name)
                    v2_ix = self.videos.index(v2_name)
                    v3_ix = self.videos.index(v3_name)
                    if (
                        self.num_frames[v1_ix] >= 32
                        and self.num_frames[v2_ix] >= 32
                        and self.num_frames[v3_ix] >= 32
                    ):
                        v1.append(v1_ix)
                        v2.append(v2_ix)
                        v3.append(v3_ix)
                    else:
                        cnt_short += 1
                        print("cnt_short {}/{}".format(cnt_short, k))
                else:
                    cnt_nonsync += 1
                    print("cnt_nonsync {}/{}".format(cnt_nonsync, k))
            self.sync_views = [v1, v2, v3]
            # 18672 * 3 = 56016 videos are in sync
            sio.savemat(cache_file, {"sync_views": self.sync_views})

    def _set_datasetname(self):
        self.datasetname = "ntu"

    def _get_video_file(self, ind):
        if self.load_res == 0.25:
            return os.path.join(
                self.img_folder, "rgb", "avi_480x270", self.videos[ind] + "_rgb.avi"
            )
        elif self.load_res == 1:
            return os.path.join(
                self.img_folder, "rgb", "avi", self.videos[ind] + "_rgb.avi"
            )
        else:
            print("Videos not pre-processed for {} resolution.".format(self.load_res))

    def _get_setup_camera_subject_repetition_action_view(self, video_name):
        setup = int(video_name[1:4])
        camera = int(video_name[5:8])
        subject = int(video_name[9:12])
        repetition = int(video_name[13:16])
        action = int(video_name[17:20]) - 1
        if camera == 1:
            view = 45
        elif (camera == 2 and repetition == 2) or (camera == 3 and repetition == 1):
            view = 0
        elif (camera == 2 and repetition == 1) or (camera == 3 and repetition == 2):
            view = 90
        return setup, camera, subject, repetition, action, view

    def _get_video_name_by_camera(self, setup, camera, subject, repetition, action):
        return "S{:03d}C{:03d}P{:03d}R{:03d}A{:03d}".format(
            setup, camera, subject, repetition, action + 1
        )

    def _get_video_name_by_view(self, setup, view, subject, repetition, action):
        if view == 45:
            camera = 1
        elif (view == 0 and repetition == 2) or (view == 90 and repetition == 1):
            camera = 2
        elif (view == 0 and repetition == 1) or (view == 90 and repetition == 2):
            camera = 3
        return "S{:03d}C{:03d}P{:03d}R{:03d}A{:03d}".format(
            setup, camera, subject, repetition, action + 1
        )

    def _get_skeleton_file(self, ind):
        return os.path.join(self.img_folder, "skeletons_mat", self.videos[ind]) + ".mat"

    def _get_action(self, ind):
        return self.actions[ind]

    def _load_joints2D(self, ind, t, kinect2smpl=True):
        if kinect2smpl:
            sub_ix = np.array((1, 17, 13, 2, 18, 14, 21, 19, 15, 21, 20, 16, 3, 9, 5, 4, 9, 5, 10, 6, 12, 8, 24, 22)) - 1
            joints2D = self.load_res * self.joints2D_kinect[0][ind][t, sub_ix, :]
        else:
            joints2D = self.load_res * self.joints2D_kinect[0][ind][t]
        return joints2D[self.jointsIx, :]

    def _load_joints3D_kinect(self, ind, frame_ix, kinect2smpl=True):
        if kinect2smpl:
            sub_ix = np.array((1, 17, 13, 2, 18, 14, 21, 19, 15, 21, 20, 16, 3, 9, 5, 4, 9, 5, 10, 6, 12, 8, 24, 22)) - 1
            joints3D = self.joints3D_kinect[0][ind][:, sub_ix, :]
        else:
            joints3D = self.joints3D_kinect[0][ind]  # (nframes, j, 3)
        joints3D = np.transpose(joints3D[frame_ix, :, :], (1, 2, 0))  # (j, 3, nframes)
        # TODO: Probably need to add this:
        joints3D[:, 2] = -joints3D[:, 2]
        # joints3D = -joints3D
        # joints3D = joints3D[:, (2, 1, 0)]
        joints3D = joints3D - joints3D[0]  # subtract pelvis
        if self.input_type == "pose":
            return to_torch(joints3D[self.jointsIx]).float().permute(1, 2, 0)
        return to_torch(joints3D[self.jointsIx].flatten()).float()  # (16, 3, nframes)

    def _load_joints3D(self, ind, frame_ix):
        """SMPL joints (hmmr/vibe)"""
        joints3D = self.joints3D_smpl[0][ind]
        if len(joints3D) == 0:
            print("Caution!")
            return torch.zeros(48)
        joints3D = np.transpose(joints3D[frame_ix, :, :], (1, 2, 0))  # (j, 3, nframes)
        joints3D[:, 1] = -joints3D[:, 1]
        joints3D[:, 2] = -joints3D[:, 2]
        joints3D = joints3D - joints3D[0]  # subtract pelvis
        if self.input_type == "pose":
            # (3, nframes, njoints
            return to_torch(joints3D[self.jointsIx]).float().permute(1, 2, 0)
        return to_torch(joints3D[self.jointsIx].flatten()).float()  # (16, 3, nframes)

    def _load_pose(self, ind, frame_ix, pose_rep="vector"):
        nframes = len(frame_ix)
        if pose_rep == "xyz":
            # SMPL joint coordinates
            return self._load_joints3D(ind, frame_ix)
        elif pose_rep == "vector":
            # SMPL axis-angle pose parameters
            if self.joints_source == "hmmr":
                pose = self.pose_smpl[0][ind][frame_ix, 3 : 3 + 72]  # (nframes, 72)
                # cam = omega[:3]
                # pose = omega[3:3 + 72]
                # shape = omega[75:]
            elif self.joints_source == "vibe":
                pose = self.pose_smpl[0][ind][frame_ix, :72]
            if self.input_type == "pose":
                # 3, nframes, 24
                pose = to_torch(pose).float().view(nframes, 3, 24).permute(1, 0, 2)
                return pose
            return to_torch(
                np.transpose(pose).flatten()
            ).float()  # (72, nframes) flatten
        elif pose_rep == "vector_noglobal":
            # SMPL axis-angle pose parameters (without the global rotation)
            if self.joints_source == "hmmr":
                pose = self.pose_smpl[0][ind][frame_ix, 6 : 6 + 69]  # (nframes, 69)
            elif self.joints_source == "vibe":
                pose = self.pose_smpl[0][ind][frame_ix, 3 : 3 + 69]
            if self.input_type == "pose":
                # (3, nframes, 23)
                pose = to_torch(pose).float().view(nframes, 3, 23).permute(1, 0, 2)
                return pose
            return to_torch(
                np.transpose(pose).flatten()
            ).float()  # (69, nframes) flatten
        elif pose_rep == "kinect_xyz":
            # Kinect joint coordinates
            return self._load_joints3D_kinect(ind, frame_ix, kinect2smpl=False)
        else:
            # Kinect joint rotations
            freejoints = kinect.free_joints()
            n_joints = freejoints.shape[0]  # 18 instead of 25
            abs_rotvec = self.pose_kinect[0][ind][frame_ix]  # (nframes, 18, 3)
            abs_rotmat = np.zeros((nframes, 25, 3, 3))
            for i in range(nframes):
                for j in range(25):
                    abs_rotmat[i, j] = utils.rotations.rotvec2rotmat(abs_rotvec[i, j])

            if pose_rep == "kinect_abs_rotmat":
                if self.input_type == "pose":
                    out = to_torch(abs_rotmat[:, freejoints, :, :])
                    out = out.view(nframes, 18, 9).float().permute(2, 0, 1)
                    return out
                return to_torch(abs_rotmat[:, freejoints, :, :].reshape(nframes, -1).transpose()).squeeze().float()

            elif pose_rep == "kinect_rotmat":
                rel_rotmat = np.zeros((nframes, 25, 3, 3))
                for i in range(nframes):
                    rel_rotmat[i] = kinect.abs2rel(abs_rotmat[i])
                return to_torch(rel_rotmat[:, freejoints, :, :].reshape(nframes, -1).transpose()).squeeze().float()

            elif pose_rep == "kinect_rotmat_noglobal":
                if self.input_type == "pose":
                    out = to_torch(abs_rotmat[:, freejoints[1:], :, :])
                    out = out.view(nframes, 17, 9).float().permute(2, 0, 1)
                    return out
                return to_torch(abs_rotmat[:, freejoints[1:], :, :].reshape(nframes, -1).transpose()).squeeze().float()
            else:
                raise ValueError(f"Pose representation {pose_rep} undefined.")
