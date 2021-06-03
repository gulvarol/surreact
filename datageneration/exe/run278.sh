#!/bin/bash
# This is how the dataset was originally created (with old Blender)
JOB_PARAMS=${1:-'--idx 7 --fend -1 --noise_factor 0.05'} # defaults

# SET PATHS HERE
FFMPEG_PATH=/sequoia/data1/gvarol/tools/ffmpeg/ffmpeg_build_sequoia_h264
X264_PATH=/sequoia/data1/gvarol/tools/ffmpeg/x264_build
BLENDER_PATH=/sequoia/data1/gvarol/tools/blender/blender-2.78a-linux-glibc211-x86_64
OPENEXR_PATH=/sequoia/data2/gvarol/tools/anaconda/envs/openexr_env3.5
CODE_PATH=/sequoia/data1/gvarol/surreact/surreact

# BUNLED PYTHON
BUNDLED_PYTHON=${BLENDER_PATH}/2.78/python
export PYTHONPATH=${BUNDLED_PYTHON}:${PYTHONPATH}

# OPENEXR PYTHON
export PYTHONPATH="${PYTHONPATH}:${OPENEXR_PATH}/lib/python3.5/site-packages"

# FFMPEG
export LD_LIBRARY_PATH=${FFMPEG_PATH}/lib:${X264_PATH}/lib:${LD_LIBRARY_PATH}
export PATH=${FFMPEG_PATH}/bin:${PATH}

# RUN BLENDER
cd ${CODE_PATH}/datageneration
$BLENDER_PATH/blender -b -t 1 -P main.py -- ${JOB_PARAMS}

