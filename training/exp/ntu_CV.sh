# Train (CV protocol)
python main.py \
    --checkpoint checkpoints/ntu/cv \
    --datasetname ntu \
    --schedule 40 45 \
    --epochs 50 \
    --num-classes 60 \
    -a rgbstream \
    --ntu_views 0-90 \
    --ntu_protocol CV \

# Test (CV protocol)
python main.py \
    --checkpoint checkpoints/ntu/cv/test \
    --datasetname ntu \
    --num-classes 60 \
    -a rgbstream \
    --ntu_views 45 \
    --ntu_protocol CV \
    --test_set test \
    -e --evaluate_video 1 \
    --pretrained checkpoints/ntu/cv/checkpoint.pth.tar \
