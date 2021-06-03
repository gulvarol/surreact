# Train (viewpoint 0) - Pose2Action
python main.py \
    --checkpoint checkpoints/ntu/v0_pose2action_vibe_pose72 \
    --datasetname ntu \
    --schedule 40 45 \
    --epochs 50 \
    --num-classes 60 \
    --ntu_views 0 \
    --input_type pose \
    -a Pose2Action \
    --num_figs 0 \
    --pose_rep vector \
    --pose_dim 24 \
    --joints_source vibe \

# Test (viewpoint 90)
python main.py \
    --checkpoint checkpoints/ntu/v0_pose2action_vibe_pose72/test90 \
    --datasetname ntu \
    --num-classes 60 \
    --ntu_views 90 \
    --test_set test \
    -e --evaluate_video 1 \
    --pretrained checkpoints/ntu/v0/checkpoint.pth.tar \
    --input_type pose \
    -a Pose2Action \
    --num_figs 0 \
    --pose_rep vector \
    --pose_dim 24 \
    --joints_source vibe \

# Other options:
# --pose_rep vector_noglobal --pose_dim 23
# --pose_rep xyz --pose_dim 24 --joints_source hmmr
# --pose_rep xyz --pose_dim 49 --joints_source vibe
# --pose_rep kinect_xyz --pose_dim 25
