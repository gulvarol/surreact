# Train (CS protocol)
python main.py \
    --checkpoint checkpoints/ntu/cs \
    --datasetname ntu \
    --schedule 40 45 \
    --epochs 50 \
    --num-classes 60 \
    -a rgbstream \
    --ntu_views 0-45-90 \

# Test (CS protocol)
python main.py \
    --checkpoint checkpoints/ntu/cs/test \
    --datasetname ntu \
    --num-classes 60 \
    -a rgbstream \
    --ntu_protocol CS \
    --test_set test \
    -e --evaluate_video 1 \
    --pretrained checkpoints/ntu/cs/checkpoint.pth.tar \
