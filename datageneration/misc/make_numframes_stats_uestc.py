import numpy as np
import sys

sys.path.append("/sequoia/data1/gvarol/similarity/render_actions")
from utils.hmmrutils import count_tracks, load_hmmr_body_data


def main():
    vidlist_path = "/home/gvarol/datasets/uestc/hmmr/subsetvideolist.txt"
    hmmr_path = "/home/gvarol/datasets/uestc/hmmr/"
    with open(vidlist_path, "r") as f:
        vid_paths = f.read().splitlines()

    num_frames = np.zeros(len(vid_paths))
    for i, name in enumerate(vid_paths):
        num_tracks, track_list = count_tracks(name, hmmr_path, datasetname="uestc")
        # assert(num_tracks == 1)
        if num_tracks == 1:
            hmmr_body_data = load_hmmr_body_data(
                name=name,
                hmmr_path=hmmr_path,
                track_id=track_list[0],
                with_trans=0,
                use_pose_smooth=0,
                datasetname="uestc",
            )
            N = len(hmmr_body_data["poses"])
            num_frames[i] = N
            print(i, N)
    import pdb

    pdb.set_trace()
    np.savetxt("uestc_hmmr_numframes.txt", num_frames, fmt="%d")


if __name__ == "__main__":
    main()
