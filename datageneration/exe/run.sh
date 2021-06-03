#!/bin/bash
# Default parameters
JOB_PARAMS=${1:-'
        --idx 7
        --fend 2
        --datasetname ntu
        --split_name train
        --smpl_estimation_method vibe
        --smpl_result_path ../data/ntu/vibe/train
        --bg_path ../data/ntu/backgrounds/
        --output_path ../data/surreact/ntu/vibe
        --tmp_path ../data/surreact/ntu/tmp_vibe_output
        --vidlist_path vidlists/ntu/train.txt '}

# SET PATHS HERE
BLENDER_PATH=/scratch/shared/beegfs/gul/surreact_project/blender-2.92.0-linux64/
CODE_PATH=/scratch/shared/beegfs/gul/surreact_project/surreact

BUNDLED_PYTHON=${BLENDER_PATH}/2.92/python
export PYTHONPATH=${BUNDLED_PYTHON}:${PYTHONPATH}
export PATH=${PATH}:${BLENDER_PATH}/2.92/python/bin

# EXTRA_ENV_PATH=/users/gul/tools/anaconda3/envs/openexr_env3.7
# export PYTHONPATH=${PYTHONPATH}:${EXTRA_ENV_PATH}/lib/python3.7/site-packages
# export PATH=${PATH}:${EXTRA_ENV_PATH}/bin
# export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${EXTRA_ENV_PATH}/lib/

cd ${CODE_PATH}/datageneration
$BLENDER_PATH/blender -b -t 1 -P main.py -- ${JOB_PARAMS}

