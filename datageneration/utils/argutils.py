import argparse
import datetime
import os
import pickle
import sys


def parse_opts():
    parser = argparse.ArgumentParser(description="Generate synth dataset images.")
    parser.add_argument(
        "--idx", type=int, default=0, help="idx of the requested sequence"
    )
    parser.add_argument(
        "--split_name", type=str, default="train", help="Options: train | test"
    )
    parser.add_argument(
        "--cam_dist",
        type=float,
        nargs="*",
        default=[5.0, 5.0],
        help="Camera distance in meters, set to [4, 6] in the paper",
    )
    parser.add_argument(
        "--cam_height",
        type=float,
        nargs="*",
        default=[1.0, 1.0],
        help="Camera height in meters, set to [-1, 3] in the paper",
    )
    parser.add_argument(
        "--zrot_euler",
        type=float,
        default=0.0,
        help="Euler rotation of the human between [0, 359]",
    )
    parser.add_argument(
        "--repetition",
        type=int,
        default=0,
        help="Repetition number for the rendering for the sequence.",
    )
    parser.add_argument(
        "--tmp_path",
        type=str,
        default="../data/surreact/ntu/tmp_vibe_output/",
        help="Path to temporary outputs which will be deleted.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../data/surreact/ntu/vibe/",
        help="Path to output folder",
    )
    parser.add_argument(
        "--bg_path",
        type=str,
        default="../data/ntu/backgrounds",
        # default='/home/gvarol/datasets/LSUN/data/img',
        help="Path to background images",
    )
    parser.add_argument(
        "--vidlist_path",
        type=str,
        default="vidlists/ntu/train.txt",
        help="Path to the list of videos.",
    )
    parser.add_argument(
        "--smpl_result_path",
        type=str,
        default="../data/ntu/vibe/train/",
        help="Path to hmmr or vibe output",
    )
    parser.add_argument(
        "--smpl_estimation_method",
        type=str,
        default="vibe",
        choices=["hmmr", "vibe"],
        help="hmmr | vibe",
    )
    parser.add_argument(
        "--use_pose_smooth", type=int, default=1, help="Temporal pose smoothing"
    )
    parser.add_argument(
        "--noise_factor",
        type=float,
        default=0.0,
        help="Additive noise range (+/-), set to 0.05 in paper experiments",
    )
    parser.add_argument(
        "--noise_level",
        type=str,
        default="video_level",
        choices=["video_level", "independent_frames", "interpolate_frames"],
        help="How to apply additive noise.",
    )
    parser.add_argument(
        "--smpl_data_folder", type=str, default="smpl_data", help="Path to smpl data"
    )
    parser.add_argument(
        "--smpl_data_filename",
        type=str,
        default="shape_params.npz",
        help="Path to the npz file with poses and trans data",
    )
    parser.add_argument(
        "--clothing_option",
        type=str,
        default="nongrey",
        help="Options: all | grey | nongrey",
    )
    parser.add_argument(
        "--with_trans", type=int, default=1, help="Whether to translate the person"
    )
    parser.add_argument(
        "--track_id",
        type=int,
        default=-1,
        help="Person/track index (all persons if -1)",
    )
    parser.add_argument("--datasetname", type=str, default="ntu", help="ntu | uestc")
    parser.add_argument("--resy", type=int, default=320, help="Width image resolution")
    parser.add_argument("--resx", type=int, default=240, help="Height image resolution")
    parser.add_argument("--fbeg", type=int, default=0, help="Beginning frame index")
    parser.add_argument(
        "--fend", type=int, default=-1, help="Ending frame index (all frames if -1)"
    )

    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


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
