# Train (viewpoint 0) - STGCN
python main.py \
    --checkpoint checkpoints/ntu/v0_stgcn_vibe_pose72 \
    --datasetname ntu \
    --schedule 40 45 \
    --epochs 50 \
    --num-classes 60 \
    --ntu_views 0 \
    --input_type pose \
    -a STGCN \
    --num_figs 0 \
    --pose_rep vector \
    --pose_dim 24 \
    --joints_source vibe \

# Test (viewpoint 90)
python main.py \
    --checkpoint checkpoints/ntu/v0_stgcn_vibe_pose72/test90 \
    --datasetname ntu \
    --num-classes 60 \
    --ntu_views 90 \
    --test_set test \
    -e --evaluate_video 1 \
    --pretrained checkpoints/ntu/v0/checkpoint.pth.tar \
    --input_type pose \
    -a STGCN \
    --num_figs 0 \
    --pose_rep vector \
    --pose_dim 24 \
    --joints_source vibe \

