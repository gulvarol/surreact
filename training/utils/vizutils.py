import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname, join
from PIL import Image
import scipy.misc
import scipy.ndimage
import torch

from utils.imutils import color_heatmap, rectangle_on_image, text_on_image
from utils.osutils import mkdir_p
import utils.disp_matplotlib as disp_matplotlib
from .misc import to_numpy


def sample_with_action(img, out_torch, action_classes=None, n=None, target=None):
    if isinstance(out_torch, torch.FloatTensor):  # Prediction
        out = torch.nn.functional.softmax(out_torch, dim=0).data
        v, out = torch.max(out, 0)
        out = out.item()
        frame_color = "green" if target[n] == out else "red"
        img = text_on_image(img, txt=action_classes[out][n])
        img = rectangle_on_image(img, frame_color=frame_color)
    else:  # Ground truth
        out = out_torch
        img = text_on_image(img, txt=action_classes[out][n])
    return img


def batch_with_action(
    inputs,
    outputs,
    mean=torch.Tensor([0.5, 0.5, 0.5]),
    std=torch.Tensor([1.0, 1.0, 1.0]),
    num_rows=2,
    parts_to_show=None,
    meta=None,
    target=None,
    save_path="",
):
    batch_img = []
    for n in range(min(inputs.size(0), 4)):
        inp = inputs[n]
        # Un-normalize
        inp = (inp * std.view(3, 1, 1).expand_as(inp)) + mean.view(3, 1, 1).expand_as(
            inp
        )
        # Torch to numpy
        inp = to_numpy(inp.clamp(0, 1) * 255)
        inp = inp.transpose(1, 2, 0).astype(np.uint8)
        # Resize 256x256 to 512x512 to be bigger
        # inp = scipy.misc.imresize(inp, [256, 256])
        batch_img.append(
            sample_with_action(
                inp,
                outputs[n],
                action_classes=meta["action_classes"],
                n=n,
                target=target,
            )
        )
    return np.concatenate(batch_img)


def visualize_gt_pred(
    inputs,
    outputs,
    target,
    mean,
    std,
    meta,
    gt_win,
    pred_win,
    fig,
    save_path=None,
    show=False,
):
    if save_path is not None:
        mkdir_p(dirname(save_path))
    # Save gif for video input
    if inputs[0].dim() == 4:
        nframes = inputs.size(2)
        save_as_gif = False
        if save_as_gif:
            gif_frames = []
        else:
            suffix = ".avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(str(save_path) + suffix, fourcc, 10, (1000, 1000))

        # For each frame
        for t in range(0, nframes, 1):
            inp = inputs[:, :, t, :, :]
            gt_win, pred_win, fig = visualize_gt_pred_single(
                inp,
                outputs,
                target,
                mean,
                std,
                meta,
                gt_win,
                pred_win,
                fig,
                save_path,
                show,
            )
            fig_img = disp_matplotlib.fig2data(fig)
            # fig_img = scipy.misc.imresize(fig_img, [1000, 1000])
            fig_img = np.array(Image.fromarray(fig_img).resize([1000, 1000]))
            if save_as_gif:
                gif_frames.append(Image.fromarray(fig_img))
            else:
                out.write(fig_img[:, :, (2, 1, 0)])
        if save_as_gif and save_path is not None:
            gif_frames[0].save(
                save_path + ".gif",
                format="GIF",
                append_images=gif_frames[1:],
                save_all=True,
                duration=100,
                loop=0,
            )
        else:
            out.release()
    return gt_win, pred_win, fig


def visualize_gt_pred_single(
    inputs, outputs, target, mean, std, meta, gt_win, pred_win, fig, save_path, show
):
    gt_batch_img = batch_with_action(
        inputs, target, mean=mean, std=std, meta=meta, save_path=dirname(save_path)
    )
    pred_batch_img = batch_with_action(
        inputs,
        outputs,
        mean=mean,
        std=std,
        meta=meta,
        target=target,
        save_path=dirname(save_path),
    )
    if not gt_win or not pred_win:
        fig = plt.figure(figsize=(20, 20))
        ax1 = plt.subplot(121)
        ax1.title.set_text("Groundtruth")
        gt_win = plt.imshow(gt_batch_img)
        ax2 = plt.subplot(122)
        ax2.title.set_text("Prediction")
        pred_win = plt.imshow(pred_batch_img)
    else:
        gt_win.set_data(gt_batch_img)
        pred_win.set_data(pred_batch_img)

    if show:
        print("Showing")
        plt.pause(0.05)

    return gt_win, pred_win, fig
