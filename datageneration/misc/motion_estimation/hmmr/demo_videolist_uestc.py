"""
Runs hmmr on a video.
Extracts tracks using AlphaPose/PoseFlow

Sample Usage:
python -m demo_video --out_dir demo_data/output
python -m demo_video --out_dir demo_data/output270k --load_path models/hmmr_model.ckpt-2699068
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from glob import glob
import json
import os.path as osp
import pickle
import re
import subprocess

from absl import flags
import ipdb
import numpy as np

from extract_tracks import compute_tracks
from src.config import get_config
from src.evaluation.run_video import (
    process_image,
    render_preds,
)
from src.evaluation.tester import Tester
from src.util.common import mkdir
from src.util.smooth_bbox import get_smooth_bbox_params

flags.DEFINE_string(
    'vid_path', 'penn_action-2278.mp4',
    'video to run on')
flags.DEFINE_integer(
    'track_id', -1,
    'PoseFlow generates a track for each detected person. This determines which'
    ' track index to use if using vid_path.'
)
flags.DEFINE_string('vidlist_path', None, 'If set, runs on all video in txt file.')
flags.DEFINE_string('out_dir', 'demo_output/',
                    'Where to save final HMMR results.')
flags.DEFINE_string('track_dir', 'demo_output/',
                    'Where to save intermediate tracking results.')
flags.DEFINE_string('pred_mode', 'pred',
                    'Which prediction track to use (Only pred supported now).')
flags.DEFINE_string('mesh_color', 'blue', 'Color of mesh.')
flags.DEFINE_integer(
    'sequence_length', 20,
    'Length of sequence during prediction. Larger will be faster for longer '
    'videos but use more memory.'
)
flags.DEFINE_boolean(
    'trim', False,
    'If True, trims the first and last couple of frames for which the temporal'
    'encoder doesn\'t see full fov.'
)


def get_labels_poseflow(json_path, num_frames, min_kp_count=20):
    """
    Returns the poses for each person tracklet.

    Each pose has dimension num_kp x 3 (x,y,vis) if the person is visible in the
    current frame. Otherwise, the pose will be None.

    Args:
        json_path (str): Path to the json output from AlphaPose/PoseTrack.
        num_frames (int): Number of frames.
        min_kp_count (int): Minimum threshold length for a tracklet.

    Returns:
        List of length num_people. Each element in the list is another list of
        length num_frames containing the poses for each person.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    if len(data.keys()) != num_frames:
        print('Not all frames have people detected in it.')
        frame_ids = [int(re.findall(r'\d+', img_name)[0])
                     for img_name in sorted(data.keys())]
        if frame_ids[0] != 0:
            print('PoseFlow did not find people in the first frame. '
                  'Needs testing.')
            # ipdb.set_trace()

    all_kps_dict = {}
    all_kps_count = {}
    for i, key in enumerate(sorted(data.keys())):
        # People who are visible in this frame.
        track_ids = []
        for person in data[key]:
            kps = np.array(person['keypoints']).reshape(-1, 3)
            idx = int(person['idx'])
            if idx not in all_kps_dict.keys():
                # If this is the first time, fill up until now with None
                all_kps_dict[idx] = [None] * i
                all_kps_count[idx] = 0
            # Save these kps.
            all_kps_dict[idx].append(kps)
            track_ids.append(idx)
            all_kps_count[idx] += 1
        # If any person seen in the past is missing in this frame, add None.
        for idx in set(all_kps_dict.keys()).difference(track_ids):
            all_kps_dict[idx].append(None)

    all_kps_list = []
    all_counts_list = []
    for k in all_kps_dict:
        if all_kps_count[k] >= min_kp_count:
            all_kps_list.append(all_kps_dict[k])
            all_counts_list.append(all_kps_count[k])

    # Sort it by the length so longest is first:
    sort_idx = np.argsort(all_counts_list)[::-1]
    all_kps_list_sorted = []
    for sort_id in sort_idx:
        all_kps_list_sorted.append(all_kps_list[sort_id])

    return all_kps_list_sorted


def predict_on_tracks(model, im_paths, all_kps, output_path, track_id,
                      trim_length):
    # Here we set which track to use.
    track_id = min(track_id, len(all_kps) - 1)
    print('Total number of PoseFlow tracks:', len(all_kps))
    print('Processing track_id:', track_id)
    kps = all_kps[track_id]

    bbox_params_smooth, s, e = get_smooth_bbox_params(kps, vis_thresh=0.1)

    images = []
    images_orig = []
    min_f = max(s, 0)
    max_f = min(e, len(kps))

    print('----------')
    print('Preprocessing frames.')
    print('----------')

    for i in range(min_f, max_f):
        proc_params = process_image(
            im_path=im_paths[i],
            bbox_param=bbox_params_smooth[i],
        )
        images.append(proc_params.pop('image'))
        images_orig.append(proc_params)

    bbox_path = osp.join(osp.dirname(output_path), 'hmmr_bbox')

    if track_id > 0:
        output_path += '_{}'.format(track_id)
        bbox_path += '_{}'.format(track_id)

    bbox_path = '{}.pkl'.format(bbox_path)
    if not osp.exists(bbox_path):
        with open(bbox_path, 'wb') as f:
            print('Saving bbox results to', bbox_path)
            pickle.dump(images_orig, f)

    mkdir(output_path)
    pred_path = osp.join(output_path, 'hmmr_output.pkl')
    if osp.exists(pred_path):
        print('----------')
        print('Loading pre-computed prediction.')
        print('----------')

        with open(pred_path, 'rb') as f:
            preds = pickle.load(f)
    else:
        print('----------')
        print('Running prediction.')
        print('----------')

        preds = model.predict_all_images(images)

        with open(pred_path, 'wb') as f:
            print('Saving prediction results to', pred_path)
            pickle.dump(preds, f)

    if trim_length > 0:
        output_path += '_trim'

    print('----------')
    print('Rendering results to {}.'.format(output_path))
    print('----------')
    render_preds(
        output_path=output_path,
        config=config,
        preds=preds,
        images=images,
        images_orig=images_orig,
        trim_length=trim_length,
    )


def run_on_video(model, vid_path, trim_length):
    """
    Main driver.
    First extracts alphapose/posetrack in track_dir
    Then runs HMMR.
    """
    print('----------')
    print('Computing tracks on {}.'.format(vid_path))
    print('----------')

    # See extract_tracks.py
    poseflow_path, img_dir = compute_tracks(vid_path, config.track_dir)

    vid_name = osp.basename(vid_path).split('.')[0]
    out_dir = osp.join(config.out_dir, vid_name, 'hmmr_output')

    # Get all the images
    im_paths = sorted(glob(osp.join(img_dir, '*.png')))
    all_kps = get_labels_poseflow(poseflow_path, len(im_paths))

    if config.track_id == -1:
        track_list = []
        # num_tracks = len(all_kps)
        num_frames = len(all_kps[0])
        # For each track
        for tid, track in enumerate(all_kps):
            cnt_nan = 0
            # For each frame
            for frame in track:
                if frame is None:
                    cnt_nan += 1
            if cnt_nan > 0:  # .1 * num_frames:
                print('Some frames ({}/{}) in track {} are None'.format(cnt_nan, num_frames, tid))
            else:
                # frame has shape 17 x 3. Take the mean over time for first 2 xy dims.
                mu = frame[:, :2].mean(axis=0)
                # The image is 960 x 540, so the person should occupy ~100 x 300
                person_box = frame[:, :2].max(axis=0) - frame[:, :2].min(axis=0)
                if person_box[1] < 200:
                    print('Person too short. Person box: {} - {}'.format(person_box, mu))
                    continue
                if mu[0] < 300 or mu[0] > 700:
                    print('Person too off. Person box: {} - {}'.format(person_box, mu))
                    continue
                print('Found tid {} with cnt_nan={}/{}. Person box: {} - {}'.format(tid, cnt_nan, num_frames, person_box, mu))
                track_list.append(tid)
                break
        # track_list = range(len(all_kps))
    else:
        track_list = [config.track_id]

    if len(track_list) == 0:
        return
        # import pdb; pdb.set_trace()

    for tid in track_list:
        additional_str = ''
        if tid > 0:
            additional_str = '_{}'.format(tid)
        mp4_path = '{}{}_crop.mp4'.format(out_dir, additional_str)
        if osp.exists(mp4_path):
            print('Already computed {} for track_id={}'.format(mp4_path, tid))
            continue
        predict_on_tracks(
            model=model,
            im_paths=im_paths,
            all_kps=all_kps,
            output_path=out_dir,
            track_id=tid,
            trim_length=trim_length
        )


def main(model):
    # Make output directory.
    mkdir(config.out_dir)

    if config.trim:
        trim_length = model.fov // 2
    else:
        trim_length = 0

    if config.vidlist_path:
        with open(config.vidlist_path, 'r') as f:
            vid_paths = f.read().splitlines()
        for vid_path in vid_paths:
            hmmr_path = '/home/gvarol/datasets/uestc/hmmr/{}'.format(vid_path[:-4])
            tmp = glob(osp.join(hmmr_path, 'hmmr_output*.pkl'))
            if len(tmp) > 0:
                print('Already extracted {}'.format(vid_path))
                continue
            vid_path = osp.join(
                '/home/gvarol/datasets/uestc/RGBvideo/',
                vid_path)
            run_on_video(model, vid_path, trim_length)
            subprocess.run(['rm', '-r', '{hmmr_path}/video_frames/'.format(hmmr_path=hmmr_path)])
    else:
        run_on_video(model, config.vid_path, trim_length)


if __name__ == '__main__':
    config = get_config()

    # Set up model:
    model_hmmr = Tester(
        config,
        pretrained_resnet_path='models/hmr_noS5.ckpt-642561'
    )

    main(model_hmmr)
