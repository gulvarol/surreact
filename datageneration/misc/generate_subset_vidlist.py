import numpy as np


def main():
    num_seq = 100
    actions = np.zeros(60)
    filename = "../vidlists/ntu/train.txt"

    newfilename = "../vidlists/train_100seq_per_action.txt"
    newf = open(newfilename, "w")

    with open(filename, "r") as f:
        videolist = f.read().splitlines()
    N = len(videolist)
    for j in range(N):
        newf.write(videolist[j])
        newf.write("\n")
        a = int(videolist[j][-3:])
        actions[a - 1] += 1
        if all(actions >= num_seq):
            exit()
            import pdb

            pdb.set_trace()


if __name__ == "__main__":
    main()
