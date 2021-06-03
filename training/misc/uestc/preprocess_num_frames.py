import cv2
import numpy as np

import datasets

loader = datasets.UESTC(img_folder="/home/gvarol/datasets/uestc/")

print("{} videos:".format(len(loader.videos)))
num_frames = []
for ind in range(len(loader.videos)):
    videofile = loader._get_video_file(ind)
    cap = cv2.VideoCapture(videofile)
    num_frames.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print("Video {} ({} frames)".format(ind, num_frames[ind]))

np.savetxt(
    "/home/gvarol/datasets/uestc/info/num_frames.txt", np.array(num_frames), fmt="%d"
)
