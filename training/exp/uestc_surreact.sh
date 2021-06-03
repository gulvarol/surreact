# Train (Real viewpoint 0 + Synth all viewpoints)
python main.py \
    --checkpoint checkpoints/uestc-surreact/synthhmmr_realv0 \
    --datasetname uestc-surreact \
    --num-classes 40 \
    --schedule 140 145 \
    --epochs 150 \
    -a rgbstream \
    --surreact_views 0-45-90-135-180-225-270-315 \
    --surreact_version uestc/hmmr \
    --uestc_train_views 0 \
    --uestc_test_views 1 \
    --watch_first_val 1 \
    --lr 1e-3 \

# Test (Real all other viewpoints)
python main.py \
    --checkpoint checkpoints/uestc-surreact/synthhmmr_realv0/test \
    --datasetname uestc \
    --num-classes 40 \
    -a rgbstream \
    --test_set test \
    -e --evaluate_video 1 \
    --uestc_train_views 1-2-3-4-5-6-7 \
    --uestc_test_views 1-2-3-4-5-6-7 \
    --pretrained checkpoints/uestc/v0/checkpoint.pth.tar \
