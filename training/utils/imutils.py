import cv2
import torch

# import torch.nn as nn
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# import random
import scipy.misc
import scipy.ndimage

from .misc import to_numpy, to_torch


def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def im_to_video(img):
    assert img.dim() == 3
    nframes = int(img.size(0) / 3)
    return img.contiguous().view(3, nframes, img.size(1), img.size(2))


def video_to_im(video):
    assert video.dim() == 4
    assert video.size(0) == 3
    return video.view(3 * video.size(1), video.size(2), video.size(3))


def load_image(img_path):
    # H x W x C => C x H x W
    return im_to_torch(scipy.misc.imread(img_path, mode="RGB"))


def resize(img, owidth, oheight):
    img = im_to_numpy(img)
    print(("%f %f" % (img.min(), img.max())))
    img = scipy.misc.imresize(img, (oheight, owidth))
    img = im_to_torch(img)
    print(("%f %f" % (img.min(), img.max())))
    return img


def resize_generic(img, oheight, owidth, interp="bilinear", is_flow=False):
    """
    Args
    inp: numpy array: RGB image (H, W, 3) | video with 3*nframes (H, W, 3*nframes)
          |  single channel image (H, W, 1) | -- not supported:  video with (nframes, 3, H, W)
    """

    # resized_image = cv2.resize(image, (100, 50))
    ht, wd, chn = img.shape[0], img.shape[1], img.shape[2]
    if chn == 1:
        resized_img = scipy.misc.imresize(
            img.squeeze(), [oheight, owidth], interp=interp, mode="F"
        ).reshape((oheight, owidth, chn))
    elif chn == 3:
        # resized_img = scipy.misc.imresize(img, [oheight, owidth], interp=interp)  # mode='F' gives an error for 3 channels
        resized_img = cv2.resize(img, (owidth, oheight))  # inverted compared to scipy
    elif chn == 2:
        # assert(is_flow)
        resized_img = np.zeros((oheight, owidth, chn), dtype=img.dtype)
        for t in range(chn):
            # resized_img[:, :, t] = scipy.misc.imresize(img[:, :, t], [oheight, owidth], interp=interp)
            # resized_img[:, :, t] = scipy.misc.imresize(img[:, :, t], [oheight, owidth], interp=interp, mode="F")
            resized_img[:, :, t] = scipy.ndimage.interpolation.zoom(
                img[:, :, t], [oheight, owidth]
            )
    else:
        in_chn = 3
        # Workaround, would be better to pass #frames
        if chn == 16:
            in_chn = 1
        if chn == 32:
            in_chn = 2
        nframes = int(chn / in_chn)
        img = img.reshape(img.shape[0], img.shape[1], in_chn, nframes)
        resized_img = np.zeros((oheight, owidth, in_chn, nframes), dtype=img.dtype)
        for t in range(nframes):
            frame = img[:, :, :, t]  # img[:, :, t*3:t*3+3]
            frame = cv2.resize(frame, (owidth, oheight)).reshape(
                oheight, owidth, in_chn
            )
            # frame = scipy.misc.imresize(frame, [oheight, owidth], interp=interp)
            resized_img[:, :, :, t] = frame
        resized_img = resized_img.reshape(
            resized_img.shape[0], resized_img.shape[1], chn
        )

    if is_flow:
        # print(oheight / ht)
        # print(owidth / wd)
        resized_img = resized_img * oheight / ht
    return resized_img


def rotate_generic(img, rot, interp="bilinear", is_flow=False):
    """
    Args
    inp: numpy array: RGB image (H, W, 3) | video with 3*nframes (H, W, 3*nframes)
         |  single channel image (H, W, 1) | -- not supported:  video with (nframes, 3, H, W)
    """

    ht, wd, chn = img.shape[0], img.shape[1], img.shape[2]
    if chn == 1:
        rotated_img = scipy.misc.imrotate(img.squeeze(), rot, interp=interp).reshape(
            (ht, wd, chn)
        )
    elif chn == 3:
        rotated_img = scipy.misc.imrotate(img, rot, interp=interp)
    else:
        nframes = int(chn / 3)
        rotated_img = np.zeros((ht, wd, chn), dtype=img.dtype)
        for t in range(nframes):
            frame = img[:, :, t * 3 : t * 3 + 3]
            frame = scipy.misc.imrotate(frame, rot, interp=interp)
            rotated_img[:, :, t * 3 : t * 3 + 3] = frame

    if is_flow:
        print("Not implemented rotating flow.")
        return
    return rotated_img


# =============================================================================
# Helpful functions generating groundtruth labelmap
# =============================================================================


def draw_labelmap(img, pt, sigma=1, type="Gaussian"):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == "Gaussian":
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == "Cauchy":
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0] : img_y[1], img_x[0] : img_x[1]] = g[g_y[0] : g_y[1], g_x[0] : g_x[1]]
    return to_torch(img)


# =============================================================================
# Helpful display functions
# =============================================================================


def gauss(x, a, b, c, d=0):
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2)) + d


def color_heatmap(x):
    x = to_numpy(x)
    color = np.zeros((x.shape[0], x.shape[1], 3))
    color[:, :, 0] = gauss(x, 0.5, 0.6, 0.2) + gauss(x, 1, 0.8, 0.3)
    color[:, :, 1] = gauss(x, 1, 0.5, 0.3)
    color[:, :, 2] = gauss(x, 1, 0.2, 0.3)
    color[color > 1] = 1
    color = (color * 255).astype(np.uint8)
    return color


def imshow(img):
    npimg = im_to_numpy(img * 255).astype(np.uint8)
    plt.imshow(npimg)
    plt.axis("off")


def show_joints(img, pts):
    imshow(img)

    for i in range(pts.size(0)):
        if pts[i, 2] > 0:
            plt.plot(pts[i, 0], pts[i, 1], "yo")
    plt.axis("off")


def rectangle_on_image(img, width=5, frame_color="yellow"):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    cor = (0, 0, img_pil.size[0], img_pil.size[1])
    for i in range(width):
        draw.rectangle(cor, outline=frame_color)
        cor = (cor[0] + 1, cor[1] + 1, cor[2] - 1, cor[3] - 1)
    return np.asarray(img_pil)


def text_on_image(img, txt=""):
    x = 5
    y = 5
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    # font = ImageFont.truetype("FreeSerif.ttf", 32)
    # font = ImageFont.truetype("DejaVuSerif.ttf", int(img.shape[0] / 8))
    font = ImageFont.load_default()
    w, h = font.getsize(txt)
    if w - 2 * x > img.shape[0]:
        font = ImageFont.truetype(
            "DejaVuSerif.ttf", int(img.shape[0] * img.shape[0] / (8 * (w - 2 * x)))
        )
        w, h = font.getsize(txt)
    draw.rectangle((x, y, x + w, y + h), fill="black")
    draw.text((x, y), txt, fill=(255, 255, 255), font=font)
    return np.asarray(img_pil)


def show_sample(inputs, target):
    num_sample = inputs.size(0)
    num_joints = target.size(1)
    height = target.size(2)
    width = target.size(3)

    for n in range(num_sample):
        inp = resize(inputs[n], width, height)
        out = inp
        for p in range(num_joints):
            tgt = inp * 0.5 + color_heatmap(target[n, p, :, :]) * 0.5
            out = torch.cat((out, tgt), 2)

        imshow(out)
        plt.show()


# =============================================================================
# Helpful functions to compute center/scale given joints2D
# =============================================================================


def tight_box_joints2D(joints2D):
    tbox = {}
    # Tighest bounding box covering the joint positions
    tbox["x_min"] = joints2D[:, 0].min()
    tbox["y_min"] = joints2D[:, 1].min()
    tbox["x_max"] = joints2D[:, 0].max()
    tbox["y_max"] = joints2D[:, 1].max()
    tbox["hum_width"] = tbox["x_max"] - tbox["x_min"] + 1
    tbox["hum_height"] = tbox["y_max"] - tbox["y_min"] + 1
    # Slightly larger area to cover the head/feet of the human
    tbox["x_min"] = tbox["x_min"] - 0.25 * tbox["hum_width"]  # left
    tbox["y_min"] = tbox["y_min"] - 0.35 * tbox["hum_height"]  # top
    tbox["x_max"] = tbox["x_max"] + 0.25 * tbox["hum_width"]  # right
    tbox["y_max"] = tbox["y_max"] + 0.25 * tbox["hum_height"]  # bottom
    tbox["hum_width"] = tbox["x_max"] - tbox["x_min"] + 1
    tbox["hum_height"] = tbox["y_max"] - tbox["y_min"] + 1
    return tbox


def get_center_joints2D(joints2D):
    tbox = tight_box_joints2D(joints2D)
    center_x = tbox["x_min"] + tbox["hum_width"] / 2
    center_y = tbox["y_min"] + tbox["hum_height"] / 2
    return [center_x, center_y]


def get_scale_joints2D(joints2D):  # imHeight
    tbox = tight_box_joints2D(joints2D)
    return max(tbox["hum_height"] / 240.0, tbox["hum_width"] / 240.0)


# =============================================================================
# Helpful functions to change segmentation indices
# =============================================================================


def change_segmix(segm, s):
    segm_new = torch.ByteTensor(segm.size()).zero_()
    for i in range(len(s)):
        segm_new[segm.eq(i + 1)] = int(s[i])
    return segm_new


def indices_to_onehot(segm):
    print("Not implemented yet.")
    return segm


# =============================================================================
# Helpful functions to manipulate partflow
# =============================================================================


def construct_partflow(segm, flow, nparts):
    """
    :param segm: Torch tensor of (H x W) size segmentation mask entries take values between [0, nparts]
    :param flow: Torch tensor of (2 x H x W) size optical flow
    :param nparts: number of parts
    :return: Torch tensor of (nparts * 2 x H x W) size part flow
    """
    assert flow.shape[0] == 2
    assert flow.shape[1] == segm.shape[0]
    assert flow.shape[2] == segm.shape[1]
    H, W = flow.shape[1], flow.shape[2]
    partflow = torch.zeros(nparts, 2, H, W)
    for i in range(nparts):
        partix = segm.eq(i)
        partflow[i][0][partix] = flow[0][partix]
        partflow[i][1][partix] = flow[1][partix]
    partflow = partflow.view(-1, H, W)
    return partflow
