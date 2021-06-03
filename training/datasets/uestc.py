import os
import numpy as np
import scipy.io as sio

from datasets.videodataset import VideoDataset
from utils.misc import to_torch


class UESTC(VideoDataset):
    def __init__(
        self,
        img_folder,
        inp_res=256,
        setname="train",
        scale_factor=0.1,
        num_in_frames=1,
        pose_rep="vector",
        evaluate_video=False,
        load_res=1,
        jointsIx=[],
        num_crops=1,
        hflip=0.5,
        train_views="1-1-1-3-3-5",
        test_views="3-5-7-5-7-7",
        randframes=False,
    ):
        print(
            "{} => Train views: {}, Test views {}".format(
                setname, train_views, test_views
            )
        )
        self.img_folder = img_folder
        self.setname = setname  # train, val or test
        self.inp_res = inp_res
        self.scale_factor = scale_factor
        self.num_in_frames = num_in_frames
        self.pose_rep = pose_rep
        self.evaluate_video = evaluate_video
        self.load_res = load_res
        self.hflip = hflip
        self.num_crops = num_crops
        self.randframes = randframes

        self.jointsIx = jointsIx

        self.img_width = int(960 * self.load_res)
        self.img_height = int(540 * self.load_res)

        with open(
            os.path.join(self.img_folder, "info", "num_frames_min.txt"), "r"
        ) as f:
            self.num_frames = np.asarray([int(s) for s in f.read().splitlines()])

        # Out of 118 subjects -> 51 training, 67 in test
        all_subjects = np.arange(1, 119)
        self.tr_subjects = [
            1, 2, 6, 12, 13, 16, 21, 24, 28, 29, 30, 31, 33, 35, 39, 41, 42, 45, 47, 50,
            52, 54, 55, 57, 59, 61, 63, 64, 67, 69, 70, 71, 73, 77, 81, 84, 86, 87, 88,
            90, 91, 93, 96, 99, 102, 103, 104, 107, 108, 112, 113]
        self.test_subjects = [s for s in all_subjects if s not in self.tr_subjects]

        # Load names of 25600 videos
        with open(os.path.join(self.img_folder, "info", "names.txt"), "r") as f:
            self.videos = f.read().splitlines()

        # Example video: a13_d3_p102_c1_color.avi
        N = len(self.videos)
        self.actions = np.zeros(N, dtype=int)  # 0-39
        self.views = np.zeros(N, dtype=int)  # 1-8
        self.subjects = np.zeros(N, dtype=int)  # 1-118
        self.sides = np.zeros(N, dtype=int)  # 1-2
        for ind in range(N):
            (
                self.actions[ind],
                self.views[ind],
                self.subjects[ind],
                self.sides[ind],
            ) = self._get_action_view_subject_side(self.videos[ind])

        # View split. 1600 each
        self.cameras = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
        # sync_views1 is always the front view and this is half of the data
        self.sync_views1 = []
        # sync_views2 is one of the 7 corners, or 8th is the moving camera
        self.sync_views2 = []
        for cam, i in self.cameras.items():
            self.sync_views1.append(
                np.where(np.logical_and(self.views == cam, self.sides == 1))[0]
            )
            self.sync_views2.append(
                np.where(np.logical_and(self.views == cam, self.sides == 2))[0]
            )

        self.view_split = [
            [elem for iterable in self.sync_views1 for elem in iterable]
        ] + self.sync_views2

        # Subject split. 0: 11360, 1: 14240
        self.subject_split = []
        self.subject_split.append(np.where(np.isin(self.subjects, self.tr_subjects))[0])
        self.subject_split.append(
            np.where(np.isin(self.subjects, self.test_subjects))[0]
        )

        self.train = []
        self.valid = []
        train_views_list = [int(x) for x in train_views.split("-")]
        test_views_list = [int(x) for x in test_views.split("-")]
        assert len(train_views_list) == len(test_views_list)

        for v in range(len(train_views_list)):
            self.train += list(self.view_split[train_views_list[v]])
            self.valid += list(self.view_split[test_views_list[v]])

        # Take a subset of 100 samples
        if self.setname == "val":
            self.valid = self.valid[:: int(len(self.valid) / 100)]
        # Redundant for now
        elif self.setname == "test":
            self.valid = self.valid

        if evaluate_video:
            self.valid, self.t_beg = self._slide_windows(self.valid, self.num_crops)

        self.rgb_pad_method = "zeros"
        self._set_action_classes()
        self._parse_what_to_load()
        VideoDataset.__init__(self)

    def _set_datasetname(self):
        self.datasetname = "uestc"

    def _get_video_file(self, ind):
        if self.load_res == 1:
            return os.path.join(self.img_folder, "RGBvideo", self.videos[ind])
        else:
            print("Videos not pre-processed for {} resolution.".format(self.load_res))

    def _get_action_view_subject_side(self, video_name):
        spl = video_name.split("_")
        action = int(spl[0][1:])
        view = int(spl[1][1:])
        subject = int(spl[2][1:])
        side = int(spl[3][1:])
        return action, view, subject, side

    def _get_video_name(self, action, view, subject, side):
        return "a{:d}_d{:d}_p{:03d}_c{:d}_color.avi".format(action, view, subject, side)

    def _get_action(self, ind):
        return self.actions[ind]

    def _load_joints2D(self, ind, t, kinect2smpl=True):
        # TODO: Check if len(joints) is smaller than num_frames
        skeleton_file = os.path.join(
            self.img_folder,
            "mat_from_skeleton",
            "{}skeleton.mat".format(self.videos[ind][:-9]),
        )
        # (num_frames, 75)
        joints3D = sio.loadmat(skeleton_file, variable_names=["v"])["v"]
        num_frames = joints3D.shape[0]
        joints3D = joints3D.reshape(num_frames, 25, 3).transpose(2, 1, 0)
        joints3D[1] *= -1  # upside down
        K = np.array(
            ((540, 0, self.img_width / 2), (0, 540, self.img_height / 2), (0, 0, 1))
        )
        joints2D = np.zeros(joints3D.shape)
        for j in range(25):
            joints2D[:, j, :] = np.dot(K, joints3D[:, j, :])
        nonzeroix = np.where(joints2D[2] != 0)
        joints2D[0][nonzeroix] = joints2D[0][nonzeroix] / joints2D[2][nonzeroix]
        joints2D[1][nonzeroix] = joints2D[1][nonzeroix] / joints2D[2][nonzeroix]
        joints2D = joints2D[:2]  # (2, 25, num_frames)
        joints2D = joints2D[:, :, t].transpose()  # (25, 2)
        if kinect2smpl:
            sub_ix = np.array((1, 17, 13, 2, 18, 14, 21, 19, 15, 21, 20, 16, 3, 9, 5, 4, 9, 5, 10, 6, 12, 8, 24, 22)) - 1
            joints2D = joints2D[sub_ix, :]  # (24, 2)
        return joints2D[self.jointsIx, :]  # (16, 2)

    def _load_joints3D(self, ind, frame_ix, kinect2smpl=True):
        skeleton_file = os.path.join(
            self.img_folder,
            "mat_from_skeleton",
            "{}skeleton.mat".format(self.videos[ind][:-9]),
        )
        joints3D = sio.loadmat(skeleton_file, variable_names=["v"])["v"]  # (T, 75)
        T = joints3D.shape[0]
        joints3D = joints3D.reshape(T, 25, 3).transpose(1, 2, 0)  # (25, 3, T)
        joints3D = joints3D[:, :, frame_ix]  # (25, 3, nframes)
        if kinect2smpl:
            sub_ix = np.array((1, 17, 13, 2, 18, 14, 21, 19, 15, 21, 20, 16, 3, 9, 5, 4, 9, 5, 10, 6, 12, 8, 24, 22)) - 1
            joints3D = joints3D[sub_ix]  # (24, 3, nframes)
        joints3D = joints3D - joints3D[0]  # subtract pelvis
        return to_torch(joints3D[self.jointsIx].flatten()).float()  # (16, 3, nframes)

    def _load_pose(self, ind, frame_ix, pose_rep="vector"):
        if pose_rep == "xyz":
            return self._load_joints3D(ind, frame_ix)
            # return {'joints3D': self._load_joints3D(ind, frame_ix)}
        else:
            print("Not defined pose representation yet")
