# Train (Real viewpoint 0 + Synth all viewpoints)
python main.py \
    --checkpoint checkpoints/ntu-surreact/synthvibe_realv0 \
    --datasetname ntu-surreact \
    --num-classes 60 \
    --schedule 140 145 \
    --epochs 150 \
    -a rgbstream \
    --surreact_views 0-45-90-135-180-225-270-315 \
    --surreact_version ntu/vibe \
    --ntu_views 0 \

# Test (Real viewpoint 90)
python main.py \
    --checkpoint checkpoints/ntu-surreact/synthvibe_realv0/test90 \
    --datasetname ntu \
    --num-classes 60 \
    -a rgbstream \
    --ntu_views 90 \
    --test_set test \
    -e --evaluate_video 1 \
    --pretrained ntu-surreact/synthvibe_realv0/checkpoint.pth.tar \
