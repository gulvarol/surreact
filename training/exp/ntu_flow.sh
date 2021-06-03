# Train (viewpoint 0) - Flow stream
python main.py \
    --checkpoint checkpoints/ntu/v0_flow \
    --datasetname ntu \
    --num-classes 60 \
    --schedule 40 45 \
    --epochs 50 \
    -a flowstream \
    --ntu_views 0 \
    --num_in_channels 2 \

# Test (viewpoint 90)
python main.py \
    --checkpoint checkpoints/ntu/v0_flow/test90 \
    --datasetname ntu \
    --num-classes 60 \
    -a flowstream \
    --ntu_views 90 \
    --test_set test \
    -e --evaluate_video 1 \
    --pretrained ntu/v0_flow/checkpoint.pth.tar \
    --num_in_channels 2 \
