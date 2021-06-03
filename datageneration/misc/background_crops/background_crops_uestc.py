import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import sys

proj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.append(proj_path)
import datasets
from utils.imutils import im_to_numpy
from utils.osutils import mkdir_p


def main():
    is_show = False
    crop_width = 320
    crop_height = 240
    setname = "train"

    # Create the root folder to store the background images
    backgrounds_root = "/home/gvarol/datasets/uestc/backgrounds/"
    backgrounds_path = os.path.join(backgrounds_root, setname)
    print("Saving background imgs to {}".format(backgrounds_path))
    if not os.path.isdir(backgrounds_path):
        mkdir_p(backgrounds_path)

    # Txt file to save the img paths
    filename = os.path.join(backgrounds_root, "{}_img.txt".format(setname))
    print("Saving background img paths to {}".format(filename))
    filetxt = open(filename, "w")

    # Data loader
    ntu_data = datasets.uestc.UESTC(
        img_folder="../../../data/uestc",
        setname=setname,
        jointsIx=np.arange(24),
        train_views="0-1-2-3-4-5-6-7",
        test_views="0-1-2-3-4-5-6-7",
    )

    # Loop over the data
    for j in range(0, len(ntu_data), 2):
        if setname == "train":
            ind = ntu_data.train[j]
        elif setname == "test":
            ind = ntu_data.valid[j]
        else:
            error("Setname {} is invalid.".format(setname))
        # print(ntu_data.videos[ind])
        T = ntu_data.num_frames[ind]
        rand_t = np.random.randint(T)
        # RGB image (3, 1080, 1920)
        rgb_full = ntu_data._load_rgb(ind, frame_ix=[rand_t])
        # (1080, 1920, 3)
        rgb_full = im_to_numpy(rgb_full)
        # 2D joints (24, 2)
        joints2D = ntu_data._load_joints2D(ind, t=rand_t)
        # 1080, 1920
        height, width = rgb_full.shape[0:2]
        # x, y positions of person bounding box
        top_left = joints2D.min(axis=0).astype(int)
        bottom_right = joints2D.max(axis=0).astype(int)

        # 4 blocks of images around the person bbox
        # Top and bottom are not very realistic
        img_background = []
        # Left
        img_background.append(rgb_full[:, 0 : top_left[0], :])
        # Right
        img_background.append(rgb_full[:, bottom_right[0] :, :])
        # Top
        # img_background.append(rgb_full[0:top_left[1], :, :])
        # Bottom
        # img_background.append(rgb_full[bottom_right[1]:, :, :])

        num_back = len(img_background)
        if is_show:
            plt.subplot(3, 2, 1)
            plt.imshow(rgb_full)
            for i in range(num_back):
                plt.subplot(3, 2, 2 + i)
                plt.imshow(img_background[i])
            plt.show()

        # For each side of the bbox (2 for left and right)
        for i in range(num_back):
            back_height, back_width = img_background[i].shape[0:2]
            new_height = int(crop_width * back_height / back_width)
            # If the side is big enough
            if back_width > 300 and new_height > 240:
                # Resize to make the width 320
                img = scipy.misc.imresize(img_background[i], [new_height, crop_width])
                # Crop random 240 from the height
                rand_h = np.random.randint(new_height - crop_height)
                img = img[rand_h : rand_h + crop_height]
                # plt.imshow(img)
                # plt.pause(0.01)
                # Save img
                filejpg_relative = "{}_{}.jpg".format(ntu_data.videos[ind], i)
                filejpg = os.path.join(backgrounds_path, filejpg_relative)
                print("{} - Saving img to {}".format(j, filejpg))
                scipy.misc.imsave(filejpg, img)
                # Save the img path to txt file
                filetxt.write("{}/{}".format(setname, filejpg_relative))
                filetxt.write("\n")
    filetxt.close()


if __name__ == "__main__":
    main()
