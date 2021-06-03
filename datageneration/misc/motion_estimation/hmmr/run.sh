cd /sequoia/data1/gvarol/others/human_dynamics/
source activate hmmr_env
export LD_LIBRARY_PATH=/sequoia/data1/gvarol/tools/ffmpeg/ffmpeg_build_sequoia_h264/lib:/sequoia/data1/gvarol/tools/ffmpeg/x264_build/lib:$LD_LIBRARY_PATH
export PATH=$PATH:/sequoia/data1/gvarol/tools/ffmpeg/ffmpeg_build_sequoia_h264/bin/

####### NTU #######
# TRAIN
python -m demo_videolist --load_path models/hmmr_model.ckpt-1119816 \
--vidlist_path /sequoia/data1/gvarol/similarity/render_actions/vidlists/ntu/train.txt \
--out_dir ~/datasets/ntu/hmmr/train/ --track_dir ~/datasets/ntu/hmmr/train/
# TEST
python -m demo_videolist --load_path models/hmmr_model.ckpt-1119816 \
--vidlist_path /sequoia/data1/gvarol/similarity/render_actions/vidlists/ntu/test_001seq_per_action.txt \
--out_dir ~/datasets/ntu/hmmr/test/ --track_dir ~/datasets/ntu/hmmr/test/

####### UESTC #######
python -m demo_videolist_uestc --load_path models/hmmr_model.ckpt-1119816 \
--vidlist_path /sequoia/data1/gvarol/similarity/render_actions/vidlists/uestc/400seq_surreact_hmmr_subset.txt \
--out_dir ~/datasets/uestc/hmmr/ --track_dir ~/datasets/uestc/hmmr/
