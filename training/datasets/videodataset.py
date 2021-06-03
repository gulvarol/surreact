import cv2
import math
import numpy as np
import os
import random

import torch
import torch.utils.data as data

from utils.imutils import (
    get_center_joints2D,
    get_scale_joints2D,
    im_to_torch,
    im_to_video,
    video_to_im,
)
from utils.misc import to_numpy, to_torch
from utils.transforms import color_normalize, crop, hflip_joints2D, im_color_jitter

cv2.setNumThreads(0)


class VideoDataset(data.Dataset):
    def __init__(self):
        self._set_meanstd()
        self._set_datasetname()
        print(
            "VideoDataset {} {} ({})".format(
                self.datasetname.upper(),
                self.setname,
                len(self),
            )
        )
        if "use_segm" in dir(self) and self.use_segm == "randbg":
            # Random video background
            if "randbgvid" in dir(self) and self.randbgvid:
                bg_file = "/home/gvarol/datasets/kinetics/mini/info/train_videos.txt"
                bg_path = "/home/gvarol/datasets/kinetics/mini/videos/"
                with open(bg_file) as f:
                    self.bg_names = f.read().splitlines()
                    self.bg_names = [os.path.join(bg_path, k) for k in self.bg_names]
            # Random image background
            else:
                bg_path = "/home/gvarol/datasets/ntu/backgrounds/"
                # bg_path = '/home/gvarol/datasets/LSUN/data/img/'
                if self.setname == "train" or self.setname == "test":
                    bg_set = self.setname
                elif self.setname == "val":
                    bg_set = "test"
                bg_file = os.path.join(bg_path, "{}_img.txt".format(bg_set))
                with open(bg_file) as f:
                    self.bg_names = f.read().splitlines()
                    self.bg_names = [os.path.join(bg_path, k) for k in self.bg_names]

    def _slide_windows(self, valid, num_crops):
        stride = int(self.num_in_frames)
        test = []
        t_beg = []
        # For each video
        for i, k in enumerate(valid):
            num_clips = (
                math.ceil((self.num_frames[k] - self.num_in_frames) / stride) + 1
            )
            # For each clip
            for j in range(num_clips):
                # Check if num_clips becomes 0
                actual_clip_length = min(
                    self.num_in_frames, self.num_frames[k] - j * stride
                )
                # For each crop
                for c in range(num_crops):
                    if actual_clip_length == self.num_in_frames:
                        t_beg.append(j * stride)
                    else:
                        t_beg.append(self.num_frames[k] - self.num_in_frames)
                    test.append(k)

        t_beg = np.asarray(t_beg)
        valid = np.asarray(test)
        return valid, t_beg

    def _get_nframes(self, ind):
        return self.num_frames[ind]

    def _set_meanstd(self):
        self.mean = 0.5 * torch.ones(3)
        self.std = 1.0 * torch.ones(3)

    def _set_action_classes(self):
        with open(
            os.path.join(self.img_folder, "info", "action_classes.txt"), "r"
        ) as f:
            self.action_classes = f.read().splitlines()

    def _load_rgb(self, ind, frame_ix, ndims=3, pad_method="zeros"):
        """
        frame_ix could be range(t, t + nframes) for consecutive reading
            or a random sorted subset of [0, video_length] of size nframes
        """
        is_consecutive = range(min(frame_ix), max(frame_ix) + 1) == frame_ix
        nframes = len(frame_ix)
        videofile = self._get_video_file(ind)
        cap = cv2.VideoCapture(videofile)
        # Do the frame setting only once if the rest are consecutive
        if is_consecutive:
            cap.set(propId=1, value=frame_ix[0])

        rgb = torch.zeros(3, nframes, self.img_height, self.img_width)
        for f, fix in enumerate(frame_ix):
            if not is_consecutive:
                cap.set(propId=1, value=fix)
            # frame: BGR, (240, 320, 3), dtype=uint8 0..255
            ret, frame = cap.read()

            if ret:
                # BGR (OpenCV) to RGB (Torch)
                frame = frame[:, :, [2, 1, 0]]
                # CxHxW (3, 240, 320), 0..1 --> np.transpose(frame, [2, 0, 1]) / 255.0
                rgb_t = im_to_torch(frame)
                rgb[:, f, :, :] = rgb_t
            else:
                if pad_method == "zeros":
                    print(f, fix, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                    print(
                        "Warning: Video frame not read (video %s, frame %d), filling zeros."
                        % (videofile, fix)
                    )
                elif pad_method == "copy_last":
                    rgb[:, f, :, :] = rgb[:, f - 1, :, :]
        cap.release()

        if ndims == 3:
            if rgb.size(0) == 1:
                rgb = rgb.squeeze()
            else:
                rgb = video_to_im(rgb)
        elif ndims == 4:
            assert rgb.dim() == 4
        else:
            print("RGB should be either 3 or 4 channels.")
        return rgb

    def _parse_what_to_load(self):
        # RGB, Joints2D, Joints3D, Segm, Flow, Pose
        self.load_info = {
            "rgb": True,
            "joints2D": True,
            "joints3D": False,
            "segm": False,
            "flow": False,
            "pose": False,
        }
        if "use_segm" in dir(self) and self.use_segm != "":
            self.load_info["segm"] = True
        if "use_flow" in dir(self) and self.use_flow != "":
            self.load_info["flow"] = True
        if hasattr(self, "input_type") and self.input_type == "pose":
            self.load_info["joints3D"] = True

        if "pose_rep" in dir(self):
            if "xyz" in self.pose_rep:
                self.load_info["joints3D"] = True

    def _get_bg_video(self, bg_filename):
        """Load random background"""
        cap_bg = cv2.VideoCapture(bg_filename)
        bg_nFrames = int(cap_bg.get(cv2.CAP_PROP_FRAME_COUNT))
        bg_width = int(cap_bg.get(cv2.CAP_PROP_FRAME_WIDTH))
        bg_height = int(cap_bg.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.randframes:
            bg_frame_ix = sorted(
                np.random.choice(range(bg_nFrames), self.num_in_frames, replace=False)
            )
        else:
            bg_t = random.randint(
                0, max(self.num_in_frames, bg_nFrames) - self.num_in_frames
            )
            bg_frame_ix = range(bg_t, bg_t + self.num_in_frames)

        bg_h = random.randint(0, max(self.img_height, bg_height) - self.img_height)
        bg_w = random.randint(0, max(self.img_width, bg_width) - self.img_width)

        # bg_video = torch.zeros(3, self.num_in_frames, bg_height, bg_width)
        bg_video = torch.zeros(3, self.num_in_frames, self.img_height, self.img_width)
        # rgb = torch.zeros(3, nframes, self.img_height, self.img_width)
        for f, fix in enumerate(bg_frame_ix):
            cap_bg.set(propId=1, value=fix)
            # frame: BGR, (240, 320, 3), dtype=uint8 0..255
            ret, frame = cap_bg.read()
            if bg_width < self.img_width or bg_height < self.img_height:
                frame = cv2.resize(frame, (self.img_width, self.img_height))
            else:
                frame = frame[
                    bg_h : bg_h + self.img_height, bg_w : bg_w + self.img_width, :
                ]
            frame = frame[:, :, [2, 1, 0]]
            # CxHxW (3, 240, 320), 0..1 --> np.transpose(frame, [2, 0, 1]) / 255.0
            bg_video[:, f, :, :] = im_to_torch(frame)
        return bg_video

    def _get_single_video(self, index, data_index, frame_ix, scale_rand, center_rand):
        # Hack for Pose2Action, return early
        if hasattr(self, "input_type") and self.input_type == "pose":
            # xyz | vector | vector_noglobal
            inp = self._load_pose(data_index, frame_ix, pose_rep=self.pose_rep)
            self.action = self._get_action(data_index)
            target = self.action

            # Meta info
            meta = {
                "index": index,
                # 'center': center,
                # 'scale': scale,
                "action_classes": self.action_classes,
                "dataset": self.datasetname,
                "action": self.action,
            }
            return inp, target, meta

        # ===================

        if self.load_info["rgb"]:
            rgb = self._load_rgb(data_index, frame_ix, pad_method=self.rgb_pad_method)
        if self.load_info["segm"]:
            # If segm is used for input
            self.segm = self._load_segm(data_index, frame_ix)
            # If segm is used for output
            # self.segm = self._load_segm(data_index, self._get_frame(t), nframes=self.num_out_frames)
        if self.load_info["flow"]:
            if "use_flow" in dir(self) and self.use_flow == "as_input":
                # If flow is used for input
                self.flow = self._load_flow(data_index, frame_ix)
            else:
                # If flow is used for 1-frame output
                self.flow = self._load_flow(data_index, [frame_ix[0] + 1])
        if self.load_info["pose"]:
            self.pose = self._load_pose(data_index, frame_ix, pose_rep=self.pose_rep)

        if "use_segm" in dir(self):
            if self.use_segm == "mask_rgb" or self.use_segm == "randbg":
                if self.use_segm == "mask_rgb":
                    fg_img = torch.zeros(
                        3, self.num_in_frames, self.img_height, self.img_width
                    )
                elif self.use_segm == "randbg":
                    if "randbgvid" in dir(self) and self.randbgvid:
                        if False:
                            class_name = self.action_classes[
                                self.actions[data_index]
                            ].split(" ")[1]
                            filtered_bg_names = [
                                b for b in self.bg_names if class_name in b
                            ]
                        else:
                            filtered_bg_names = self.bg_names
                        bg_filename = np.random.choice(filtered_bg_names)
                        fg_img = self._get_bg_video(bg_filename)
                    else:
                        bg_filename = np.random.choice(self.bg_names)
                        bg_img_cv2 = cv2.imread(bg_filename)
                        bg_img_cv2 = cv2.resize(
                            bg_img_cv2, (self.img_width, self.img_height)
                        )
                        # 3, 240, 320 (RGB)
                        bg_img = im_to_torch(bg_img_cv2)[(2, 1, 0), :, :]  
                        fg_img = bg_img.view(
                            3, 1, self.img_height, self.img_width
                        ).repeat(1, self.num_in_frames, 1, 1)
                rgb = im_to_video(rgb)
                fg_ix = self.segm.ne(0)
                fg_img[:, fg_ix] = rgb[:, fg_ix]
                rgb = video_to_im(fg_img)
            elif self.use_segm == "as_input":
                rgb = np.zeros((self.num_in_frames, self.img_height, self.img_width, 3))
                for f in range(self.num_in_frames):
                    rgb[f] = cv2.applyColorMap(
                        to_numpy(self.segm[f] * 255 / 15).astype("uint8"),
                        cv2.COLORMAP_JET,
                    )
                rgb = (
                    video_to_im(
                        to_torch(rgb.transpose((3, 0, 1, 2))).contiguous()
                    ).float()
                    / 255.0
                )

        if "use_flow" in dir(self):
            if self.use_flow == "as_input":
                flow_proc = self.flow.clamp(-30, 30).div(60.0) + 0.5
                mag = torch.sqrt(flow_proc[0] ** 2 + flow_proc[1] ** 2)
                mag.div_(math.sqrt(2))
                flow_proc = torch.cat((flow_proc, mag.unsqueeze(0)), dim=0)
                # print('before', rgb.min(), rgb.mean(), rgb.max())
                rgb = video_to_im(flow_proc)
                # Note: changing rgb changes flow_proc

        self.joints2D = torch.FloatTensor(self._load_joints2D(data_index, frame_ix[0]))
        scale = get_scale_joints2D(self.joints2D)
        center = torch.FloatTensor(get_center_joints2D(self.joints2D))
        if (
            center[0] < 1
            or center[1] < 1
            or center[0] > self.img_width
            or center[1] > self.img_height
        ):
            # print('\nHuman center out of range [%d %d].\n' % (center[0], center[1]))
            center = torch.FloatTensor([self.img_width / 2, self.img_height / 2])
            scale = torch.tensor(1.0)

        if self.setname == "train":
            # Horizontal flip
            if random.random() < self.hflip:
                rgb = torch.flip(rgb, dims=[2])
                if "pose" in dir(self):
                    assert self.pose_rep == "xyz"
                    self.pose = hflip_joints3D(self.pose.reshape(16, 3)).flatten()
                if "joints2D" in dir(self):
                    # Hflipping only defined for 16-d, not important if the supervision is not pose
                    # Do this before cropping, because we use the full img_width
                    self.joints2D = hflip_joints2D(self.joints2D, width=self.img_width)
                center[0] = self.img_width - center[0]
                if "flow" in dir(self):
                    print("Flow hflipping not implemented.")
                if "segm" in dir(self):
                    print("Segm hflipping not implemented.")
            # Color jitter if the input is not flow
            if not ("use_flow" in dir(self) and self.use_flow == "as_input"):
                rgb = im_color_jitter(rgb, num_in_frames=self.num_in_frames, thr=0.2)

        rot = 0
        # Jitter at train or when evaluating with num_crops > 1
        if self.setname == "train" or (self.num_crops > 1 and self.evaluate_video):
            # Jitter box
            scale = scale * scale_rand
            center_offsets = scale * center_rand
            center = center + center_offsets

        inp = crop(rgb, center, scale, [self.inp_res, self.inp_res], rot=rot)

        if self.num_in_frames > 1:
            inp = im_to_video(inp)

        inp = color_normalize(inp, self.mean, self.std)

        if self.num_in_frames == 2:
            inp = np.transpose(inp, (1, 0, 2, 3))  # nframes, 3, height, width
            inp = inp.reshape(6, inp.shape[2], inp.shape[3])

        if (
            self.load_info["segm"]
            and self.use_segm != "mask_rgb"
            and self.use_segm != "randbg"
        ):
            self.segm = crop(
                self.segm.view(-1, self.img_height, self.img_width),
                center,
                scale,
                [self.out_res, self.out_res],
                rot=rot,
                interp="nearest",
                rgb=False,
            )
            self.segm = self.segm.long().squeeze()
        if self.load_info["flow"]:
            self.flow = crop(
                self.flow.view(-1, self.img_height, self.img_width),
                center,
                scale,
                [self.out_res, self.out_res],
                rot=rot,
                interp="bilinear",
                rgb=False,
                is_flow=True,
            )

        self.action = self._get_action(data_index)
        target = self.action

        # Meta info
        meta = {
            "index": index,
            "center": center,
            "scale": scale,
            "action_classes": self.action_classes,
            "dataset": self.datasetname,
            "action": self.action,
        }

        # if '_get_camera_angle' in dir(self):
        #     meta['camera_angle'] = self._get_camera_angle(data_index)

        return inp, target, meta

    def __getitem__(self, index):
        if self.setname == "train":
            data_index = self.train[index]
        else:
            data_index = self.valid[index]

        # Number of frames
        nFrames = self._get_nframes(data_index)

        scale_rand = 1
        center_rand = 0

        # TEST
        if self.evaluate_video:
            # Pre-computed frame number for sliding window testing
            t = self.t_beg[index]
            if self.num_crops > 1 and self.evaluate_video:
                scale_rand = (
                    torch.randn(1)
                    .mul_(self.scale_factor)
                    .add_(1)
                    .clamp(1 - self.scale_factor, 1 + self.scale_factor)[0]
                )
                center_rand = torch.rand(2) * self.img_height * 0.02 - 1
        else:
            # TRAIN
            if self.setname == "train":
                # Random cropping
                scale_rand = (
                    torch.randn(1)
                    .mul_(self.scale_factor)
                    .add_(1)
                    .clamp(1 - self.scale_factor, 1 + self.scale_factor)[0]
                )
                center_rand = torch.rand(2) * self.img_height * 0.05 - 1
                # Random frame number for training
                # If the video has less than num_in_frames frames, t=0
                if self.load_info["flow"]:
                    # starting from 2nd frame (1st flow not meaningful)
                    t = random.randint(
                        1, max(self.num_in_frames, nFrames) - self.num_in_frames
                    )
                else:
                    t = random.randint(
                        0, max(self.num_in_frames, nFrames) - self.num_in_frames
                    )
            # VAL
            else:
                # Middle frame for val
                # If the video has less than num_in_frames frames, t=0
                if nFrames < self.num_in_frames * 2:
                    t = 0
                else:
                    t = math.ceil(nFrames / 2)

        # Define if consecutive or randomly sampled frames
        if self.randframes:  # and not self.evaluate_video:
            # HACK
            hybrid = False
            uniform = False
            randfps = False
            if hybrid:
                segments = (
                    np.linspace(0, nFrames, self.num_in_frames + 1).round().astype(int)
                )
                frame_ix = []
                for i in range(len(segments) - 1):
                    frame_ix.append(
                        np.random.choice(range(segments[i], segments[i + 1]), 1)[0]
                    )
            elif uniform:
                segments = (
                    np.linspace(0, nFrames, self.num_in_frames + 1).round().astype(int)
                )
                frame_ix_within_segment = np.random.randint(segments[1])
                frame_ix = []
                for i in range(len(segments) - 1):
                    frame_ix.append(segments[i] + frame_ix_within_segment)
            elif randfps:
                # Select the number of frames to sample (determines the timescale)
                # Sample something between the num_in_frames and the full span.
                # Revert to num_in_frames in case the full span is smaller.
                # In that case, the video will be slowed down, repeating frames.
                num_scaled_frames = random.randint(
                    self.num_in_frames, max(self.num_in_frames, nFrames)
                )
                # Shrink/stretch the video with a constant scale
                scaled_frames = np.linspace(0, nFrames - 1, num_scaled_frames)
                scaled_frames = scaled_frames.round().astype(int)
                beg = random.randint(0, max(0, len(scaled_frames) - self.num_in_frames))
                frame_ix = scaled_frames[beg : beg + self.num_in_frames].tolist()
            else:
                frame_ix = sorted(
                    np.random.choice(range(nFrames), self.num_in_frames, replace=False)
                )
            # frame_ix = np.random.choice(range(nFrames), self.num_in_frames, replace=False)
        else:
            frame_ix = range(t, t + self.num_in_frames)

        inp1, target1, meta1 = self._get_single_video(
            index, data_index, frame_ix, scale_rand, center_rand
        )
        return inp1, target1, [meta1]

    def __len__(self):
        if self.setname == "train":
            return len(self.train)
        else:
            return len(self.valid)
