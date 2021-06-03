import os
import random
from random import choice


def pick_background(bg_path, split_name):
    bg_names = os.path.join(bg_path, "{}_img.txt".format(split_name))
    bg_paths = []
    with open(bg_names) as f:
        for line in f:
            bg_paths.append(os.path.join(bg_path, line))
    bg_img_name = choice(bg_paths)[:-1]
    # Original
    # bg_img_name = '/home/gvarol/datasets/ntu/backgrounds/train/S003C003P019R001A021_1.jpg'
    # Different background
    # bg_img_name = '/home/gvarol/datasets/ntu/backgrounds/train/S007C002P017R002A010_1.jpg'
    print("Picked bg file: {}".format(bg_img_name))
    return bg_img_name


def pick_cloth(clothing_option, smpl_data_folder, split_name):
    # grab clothing names from female/male genders
    print("clothing: {}".format(clothing_option))
    with open(
        os.path.join(smpl_data_folder, "textures", "female_{}.txt".format(split_name))
    ) as f:
        txt_paths = f.read().splitlines()
    with open(
        os.path.join(smpl_data_folder, "textures", "male_{}.txt".format(split_name))
    ) as f:
        txt_paths += f.read().splitlines()

    # if using only one source of clothing
    if clothing_option == "nongrey":
        txt_paths = [k for k in txt_paths if "nongrey" in k]
    elif clothing_option == "grey":
        txt_paths = [k for k in txt_paths if "nongrey" not in k]
    elif clothing_option == "same":
        # txt_paths = ['textures/female/nongrey_female_0010.jpg']
        # Orig
        txt_paths = ["textures/male/nongrey_male_0244.jpg"]
        # Different texture
        # txt_paths = ['textures/male/nongrey_male_0529.jpg']

    # random clothing texture
    cloth_img_name = choice(txt_paths)
    cloth_img_name = os.path.join(smpl_data_folder, cloth_img_name)
    print("Picked clothing file: {}".format(cloth_img_name))
    return cloth_img_name


def pick_shape(smpl_data, gender, split_name):
    fshapes = smpl_data["{}shapes".format(gender)]

    nb_fshapes = len(fshapes)
    if split_name == "train":
        fshapes = fshapes[: int(nb_fshapes * 0.8)]
    elif split_name == "test":
        fshapes = fshapes[int(nb_fshapes * 0.8) :]
    else:
        print(
            "Split "
            "{}"
            " is not a valid argument. Options: train, test.".format(split_name)
        )
        exit()
    # pick random real body shape
    shape = choice(fshapes)

    # import numpy as np
    # Fat
    # shape = np.array([0, -3.7883464, 0.46747496, 3.89178988, 2.20098416, 0.26102114, -3.07428093, 0.55708514, -3.94442258, -2.88552087])
    # Average
    # shape = np.zeros(10)
    return shape


def pick_cam(cam_height_range, cam_dist_range):
    cam_height = random.uniform(cam_height_range[0], cam_height_range[1])
    cam_dist = random.uniform(cam_dist_range[0], cam_dist_range[1])
    return cam_height, cam_dist
