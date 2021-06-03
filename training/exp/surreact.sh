# Train
python main.py \
    --checkpoint checkpoints/surreact/vibe \
    --datasetname surreact \
    --num-classes 60 \
    --schedule 60 65 \
    --epochs 70 \
    -a rgbstream \
    --surreact_views 0-45-90-135-180-225-270-315 \
    --surreact_version ntu/vibe \

# Test
python main.py \
    --checkpoint checkpoints/surreact/vibe/test \
    --datasetname surreact \
    --num-classes 60 \
    -a rgbstream \
    --surreact_views 0-45-90-135-180-225-270-315 \
    --surreact_version ntu/vibe \
    --test_set test \
    -e --evaluate_video 1 \
    --pretrained checkpoints/surreact/vibe/checkpoint.pth.tar \
