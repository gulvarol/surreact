import argparse
import collections
import datetime
import models
import os
import pickle

model_names = sorted(
    name
    for name in models.__dict__
    # if name.islower() and not name.startswith("__")
    if not name.startswith("__")
    and isinstance(models.__dict__[name], collections.Callable)
)


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    # Model structure
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="model3D",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names),
    )
    parser.add_argument(
        "--num-classes", default=60, type=int, metavar="N", help="Number of classes"
    )
    # Training strategy
    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "--train-batch", default=10, type=int, metavar="N", help="train batchsize"
    )
    parser.add_argument(
        "--test-batch", default=3, type=int, metavar="N", help="test batchsize"
    )
    parser.add_argument(
        "--optim", default="rmsprop", type=str, help="Options: rmsprop | sgd | adam"
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=1e-3,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--momentum", default=0, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=0,
        type=float,
        metavar="W",
        help="weight decay (default: 0)",
    )
    parser.add_argument(
        "--schedule",
        type=int,
        nargs="+",
        default=[40, 45],
        help="Decrease learning rate at these epochs.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="LR is multiplied by gamma on schedule.",
    )
    # Miscs
    parser.add_argument(
        "--snapshot", default=5, type=int, metavar="N", help="frequency of saving model"
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default="checkpoint",
        type=str,
        metavar="PATH",
        help="path to save checkpoint (default: checkpoint)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on test set",
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="debug",
        action="store_true",
        help="show intermediate results",
    )
    parser.add_argument(
        "--inp_res",
        type=int,
        default="256",
        help="Spatial resolution of the network input.",
    )
    parser.add_argument(
        "--num_in_frames", type=int, default="16", help="Number of input frames."
    )
    parser.add_argument(
        "--num_in_channels",
        type=int,
        default="3",
        help="Number of input channels (set to 2 for flowstream).",
    )
    parser.add_argument(
        "--datasetname",
        type=str,
        default="surreal",
        help="surreal | ntu | nucla | hri40 | nuclantu | ntusurreal",
    )
    parser.add_argument(
        "--pretrained", type=str, default="", help="path to pretrained model file"
    )
    parser.add_argument(
        "--evaluate_video",
        type=int,
        default="0",
        help="whether to test on sliding windows",
    )
    parser.add_argument(
        "--test_set",
        type=str,
        default="val",
        help="Which set to evaluate on: val | test",
    )
    parser.add_argument(
        "--ntu_protocol",
        type=str,
        default="",
        help="Train/split definition of NTU dataset: CV | CS | " "",
    )
    parser.add_argument(
        "--pose_rep", type=str, default="vector", help="SMPL pose representation"
    )
    parser.add_argument("--with_dropout", type=float, default=0.0, help="Dropout value")
    parser.add_argument(
        "--pose_dim",
        type=int,
        default=24,
        help="Number of joints in the rotation angles (24 for SMPL, 18 for Kinect)",
    )
    parser.add_argument(
        "--joints_source", type=str, default="hmmr", help="hmmr | vibe | (kinect)"
    )
    parser.add_argument(
        "--ntu_views",
        type=str,
        default="0",
        help="viewpoints for NTU dataset (0 | 45 | 90 | 0-45 | 0-90 | 0-45-90)",
    )
    parser.add_argument(
        "--uestc_train_views",
        type=str,
        default="0",
        help="viewpoints for UESTC training (0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 and their combinations with dash)",
    )
    parser.add_argument(
        "--uestc_test_views",
        type=str,
        default="1",
        help="viewpoints for UESTC test (0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 and their combinations with dash)",
    )
    parser.add_argument(
        "--surreact_views",
        type=str,
        default="0",
        help="viewpoints for SURREACT  (0 | 45 | 90 | 135 | 180 | 225 | 270 | 315 and their combinations with dash)",
    )
    parser.add_argument(
        "--nloss", type=int, default=1, help="number of losses to keep track of"
    )
    parser.add_argument(
        "--nperf", type=int, default=1, help="number of performance metrics"
    )
    parser.add_argument(
        "--num_figs",
        type=int,
        default=10,
        help="controls frequency to save figures (default 10)",
    )
    parser.add_argument(
        "--surreact_version",
        type=str,
        default="ntu/vibe",
        help="Version of the surreact dataset.",
    )
    parser.add_argument(
        "--surreact_matfile",
        type=str,
        default="surreact_data.mat",
        help="Version of the surreact dataset.",
    )
    parser.add_argument(
        "--use_segm",
        type=str,
        default="",
        help="For surreact: - | as_input | mask_rgb | randbg",
    )
    parser.add_argument(
        "--randbgvid",
        type=int,
        default=0,
        help="For surreact: whether to use images or videos as rand background.",
    )
    parser.add_argument(
        "--use_flow", type=str, default="", help="For surreact: - | as_input"
    )
    parser.add_argument(
        "--randframes",
        type=int,
        default=1,
        help="Whether to sample random frames or consecutive.",
    )
    parser.add_argument(
        "--watch_first_val",
        type=int,
        default=0,
        help="Watch the validation performance of the first dataset of the mix.",
    )
    parser.add_argument(
        "--input_type", type=str, default="rgb", help="Options: rgb | pose"
    )

    return parser.parse_args()


def print_args(args):
    print("==== Options ====")
    for k, v in sorted(vars(args).items()):
        print("{}: {}".format(k, v))
    print("=================")


def save_args(args, save_folder, opt_prefix="opt", verbose=True):
    opts = vars(args)
    os.makedirs(save_folder, exist_ok=True)

    # Save to text
    opt_filename = "{}.txt".format(opt_prefix)
    opt_path = os.path.join(save_folder, opt_filename)
    with open(opt_path, "a") as opt_file:
        opt_file.write("====== Options ======\n")
        for k, v in sorted(opts.items()):
            opt_file.write("{option}: {value}\n".format(option=str(k), value=str(v)))
        opt_file.write("=====================\n")
        opt_file.write("launched at {}\n".format(str(datetime.datetime.now())))

    # Save as pickle
    opt_picklename = "{}.pkl".format(opt_prefix)
    opt_picklepath = os.path.join(save_folder, opt_picklename)
    with open(opt_picklepath, "wb") as opt_file:
        pickle.dump(opts, opt_file)
    if verbose:
        print("Saved options to {}".format(opt_path))
