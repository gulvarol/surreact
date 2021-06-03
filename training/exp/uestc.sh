# Train (viewpoint 0)
python main.py \
    --checkpoint checkpoints/uestc/v0 \
    --datasetname uestc \
    --schedule 40 45 \
    --epochs 50 \
    --num-classes 40 \
    -a rgbstream \
    --uestc_train_views 0 \
    --uestc_test_views 1 \

# Test (all other viewpoints)
python main.py \
    --checkpoint checkpoints/uestc/v0/test \
    --datasetname uestc \
    --num-classes 40 \
    -a rgbstream \
    --test_set test \
    -e --evaluate_video 1 \
    --uestc_train_views 1-2-3-4-5-6-7 \
    --uestc_test_views 1-2-3-4-5-6-7 \
    --pretrained checkpoints/uestc/v0/checkpoint.pth.tar \
