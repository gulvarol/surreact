# Example standalone usage:
# python evaluate.py --datasetname ntu --checkpoint checkpoints/ntu/cvsp/test45/ --ntu_views 45
# python evaluate.py --datasetname uestc --checkpoint checkpoints/uestc/odd/test_even/ --uestc_train_views 1-3-5-7 --uestc_test_views 0-2-4-6
# python evaluate.py --datasetname surreact --checkpoint checkpoints/ntu/cvsp/testsynth/ --surreact_version ntu/hmmr --surreact_views 0-45-90-135-180-225-270-315
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import scipy.io as sio
import os

import opts
import datasets

plt.switch_backend("agg")


def aggregate_clips(dataloader_val, scores):
    video_ix = np.unique(dataloader_val.valid)
    N = len(video_ix)
    gt_action = np.zeros(N)
    pred_action = np.zeros(N)
    len_clip = np.zeros(N)

    if scores is not None:
        vid_scores = np.zeros((N, scores.shape[1]))
    else:
        vid_scores = None
    if isinstance(dataloader_val, datasets.surreact.SURREACT):
        view = np.zeros(N)

    for i, vid in enumerate(video_ix):
        clip_ix = np.where(dataloader_val.valid == vid)
        if scores is not None:
            clip_score = scores[clip_ix]
            len_clip[i] = clip_score.shape[0]
            vid_score = np.mean(clip_score, axis=0)
            pred_action[i] = np.argmax(vid_score)
            gt_action[i] = dataloader_val.actions[vid]
        if isinstance(dataloader_val, datasets.surreact.SURREACT):
            view[i] = dataloader_val.views[vid]
    if isinstance(dataloader_val, datasets.hri40.HRI40):
        ix = []
        for v in range(8):
            ix = np.where(np.isin(video_ix, dataloader_val.view_split[v]))[0]
            if len(ix) == 0:
                acc = 0
            else:
                acc, confmat, nc, na = get_acc(gt_action[ix], pred_action[ix])
            print("V{}: {:.1f} ({})".format(v, acc, len(ix)))
    if isinstance(dataloader_val, datasets.surreact.SURREACT):
        ix = []
        for v in range(0, 360, 45):
            ix = v == view
            if len(ix) == 0:
                acc = 0
            else:
                acc, confmat, nc, na = get_acc(gt_action[ix], pred_action[ix])
            print("V{}: {:.1f} ({})".format(v, acc, sum(ix)))
            print(np.diag(confmat))

    return gt_action, pred_action, len_clip, vid_scores


def get_acc(gt, pred):
    num_correct = (gt == pred).sum()
    num_all = len(gt)
    accuracy = 100 * num_correct / num_all
    confmat = sklearn.metrics.confusion_matrix(gt, pred)
    return accuracy, confmat, num_correct, num_all


def viz_confmat(confmat, accuracy, categories, save_path=None):
    # shorten category names
    for i, c in enumerate(categories):
        if len(c) > 20:
            categories[i] = c[:20]
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(confmat)
    # plt.colorbar()
    plt.xticks(np.arange(len(categories)), categories, rotation="vertical", fontsize=8)
    plt.yticks(np.arange(len(categories)), categories, fontsize=8)
    plt.xlabel("predicted labels")
    plt.ylabel("true labels")
    plt.title("Accuracy = {:.1f}".format(accuracy))
    if save_path:
        plt.savefig(save_path)
    plt.close()


def evaluate(dataloader_val, exp):
    exp_root = "{exp}".format(exp=exp)
    mat_file = "{exp_root}/preds_valid.mat".format(exp_root=exp_root)
    print("Loading from {}".format(mat_file))
    experiment = sio.loadmat(mat_file)
    pred_scores = experiment["preds"]
    print(pred_scores.shape)  # e.g. [32558, 60]
    assert pred_scores.shape[0] == dataloader_val.valid.shape[0]
    gt, pred, len_clip, vid_scores = aggregate_clips(dataloader_val, scores=pred_scores)
    accuracy, confmat, num_correct, num_all = get_acc(gt, pred)

    # Save to be able to reproduce
    results_file = "{exp_root}/results.mat".format(exp_root=exp_root)
    print("Saving to {}".format(results_file))
    results_dict = {
        "accuracy": accuracy,
        "confmat": confmat,
        "num_correct": num_correct,
        "num_all": num_all,
        "gt": gt,
        "pred": pred,
        "test_index": dataloader_val.valid,
        "test_t_beg": dataloader_val.t_beg,
        "videos": dataloader_val.videos,
    }
    sio.savemat(results_file, results_dict)

    # Print to file for better readability
    with open(results_file.replace(".mat", ".txt"), "w") as f:
        f.write("Histogram of clip lengths ({} clips):\n".format(pred_scores.shape[0]))
        f.write(
            np.array2string(np.histogram(len_clip, bins=np.unique(len_clip))[0])
            + "\n\n"
        )
        f.write("Histogram of GT - Pred per action:\n")
        for i in range(len(np.unique(gt))):
            f.write("{}: {} - {}\n".format(i, (gt == i).sum(), (pred == i).sum()))
        f.write("Confusion matrix:\n")
        f.write(
            np.array2string(confmat, threshold=np.nan, max_line_width=np.nan) + "\n\n"
        )
        f.write("Accuracy: {:.2f}% ({}/{})\n".format(accuracy, num_correct, num_all))

    # Std out
    print("Histogram of clip lengths ({} clips):".format(pred_scores.shape[0]))
    print(np.histogram(len_clip, bins=np.unique(len_clip))[0])
    print("Histogram of GT - Pred per action:\n")
    for i in range(len(np.unique(gt))):
        print("{}: {} - {}".format(i, (gt == i).sum(), (pred == i).sum()))
    print("Confusion matrix:")
    print(confmat)
    print("Accuracy: {:.2f}% ({}/{})".format(accuracy, num_correct, num_all))
    viz_confmat(
        confmat,
        accuracy,
        dataloader_val.action_classes,
        save_path="{exp_root}/confmat.png".format(exp_root=exp_root),
    )
    return pred_scores, accuracy


def get_dataloader(args):
    if args.datasetname == "ntu":
        loader = datasets.NTU(
            "data/ntu/info/ntu_data.mat",
            "data/ntu",
            num_in_frames=args.num_in_frames,
            pose_rep=args.pose_rep,
            inp_res=args.inp_res,
            setname=args.test_set,
            evaluate_video=args.evaluate_video,
            protocol=args.ntu_protocol,
            views_list_str=args.ntu_views,
            input_type=args.input_type,
            joints_source=args.joints_source,
        )
    elif args.datasetname == "uestc":
        loader = datasets.UESTC(
            "data/uestc",
            num_in_frames=args.num_in_frames,
            inp_res=args.inp_res,
            setname=args.test_set,
            evaluate_video=args.evaluate_video,
            train_views=args.uestc_train_views,
            test_views=args.uestc_test_views,
        )
    elif args.datasetname == "surreact":
        loader = datasets.SURREACT(
            "data/surreact/{}".format(args.surreact_version),
            matfile=args.surreact_matfile,
            inp_res=args.inp_res,
            num_in_frames=args.num_in_frames,
            pose_rep=args.pose_rep,
            setname=args.test_set,
            evaluate_video=args.evaluate_video,
            views_list_str=args.surreact_views,
            randframes=args.randframes,
            use_segm=args.use_segm,
            use_flow=args.use_flow,
        )
    else:
        raise ValueError(f"Undefined dataset {args.datasetname}.")
    return loader


if __name__ == "__main__":
    args = opts.parse_opts()
    # args.num_in_frames = 16
    args.evaluate_video = 1
    args.test_set = "test"
    dataloader_val = get_dataloader(args)
    evaluate(dataloader_val, exp=args.checkpoint)
