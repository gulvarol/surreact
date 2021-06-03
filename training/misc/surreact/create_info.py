import cv2
import os
import scipy.io as sio
import shutil
from tqdm import tqdm


def main():
    root_path = "/home/gvarol/datasets/surreact/ntu/"
    version_name = "vibe"
    version_path = os.path.join(root_path, version_name)
    dict_data = {}
    dict_data["names"] = []
    dict_data["T"] = []
    dict_data["splits"] = [[], []]
    ind = 0
    # For each split
    for setname in ["train", "test"]:
        set_path = os.path.join(version_path, setname)
        # For each sequence
        for seqname in tqdm(os.listdir(set_path)):
            seq_path = os.path.join(set_path, seqname)
            # For each viewpoint
            for videoname in os.listdir(seq_path):
                segm_file = os.path.join(seq_path, videoname[:-4] + "_segm.mat")
                if ".mp4" in videoname and os.path.isfile(segm_file):
                    video_path = os.path.join(seq_path, videoname)
                    dict_data["names"].append(videoname)
                    if setname == "train":
                        dict_data["splits"][0].append(ind)
                    elif setname == "test":
                        dict_data["splits"][1].append(ind)
                    cap = cv2.VideoCapture(video_path)
                    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    dict_data["T"].append(num_frames)
                    ind += 1
                    print(ind)
    info_path = os.path.join(version_path, "info")
    if not os.path.isdir(info_path):
        os.makedirs(info_path)
    matfile = os.path.join(info_path, "surreact_data.mat")
    print(
        "{} training, {} testing.".format(
            len(dict_data["splits"][0]), len(dict_data["splits"][1])
        )
    )
    print("Saving info to {}".format(matfile))
    sio.savemat(matfile, dict_data)

    # Copy action_classes file from the first version into info folder
    shutil.copyfile(
        os.path.join(root_path, "v01", "info", "action_classes.txt"),
        os.path.join(version_path, "info", "action_classes.txt"),
    )


if __name__ == "__main__":
    main()
