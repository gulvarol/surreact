# Train (viewpoint 0)
python main.py \
    --checkpoint checkpoints/ntu/v0 \
    --datasetname ntu \
    --schedule 40 45 \
    --epochs 50 \
    --num-classes 60 \
    -a rgbstream \
    --ntu_views 0 \

# Test (viewpoint 90)
python main.py \
    --checkpoint checkpoints/ntu/v0/test90 \
    --datasetname ntu \
    --num-classes 60 \
    -a rgbstream \
    --ntu_views 90 \
    --test_set test \
    -e --evaluate_video 1 \
    --pretrained checkpoints/ntu/v0/checkpoint.pth.tar \
