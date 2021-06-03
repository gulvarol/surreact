import pickle as pkl
import scipy.io as sio

dict_data = sio.loadmat(
    "/home/gvarol/datasets/surreact/uestc/vibe/info/surreact_data.mat", squeeze_me=True
)

num_frames = dict_data["T"]
videos = [s.strip() for s in dict_data["names"]]

for i in range(len(num_frames)):
    videofile = (
        f"/home/gvarol/datasets/surreact/uestc/vibe/train/{videos[i][:-13]}/{videos[i]}"
    )
    infofile = videofile[:-4] + "_info.mat"
    joints2D = sio.loadmat(infofile, variable_names=["joints2D"])["joints2D"]
    if num_frames[i] > joints2D.shape[3]:
        print(num_frames[i], joints2D.shape)
        num_frames[i] = joints2D.shape[3]

num_frames_pkl = "/home/gvarol/datasets/surreact/uestc/vibe/info/num_frames.pkl"
pkl.dump(num_frames, open(num_frames_pkl, "wb"))
