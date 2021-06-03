# CAUTION:
# This script is very likely to fail at some point (especially as the versions get outdated),
# it is intended mainly to provide a step by step guide
# for setting up the prerequisites to run the data generation code.
# Please run it at your own risk and know that there might not be support in case of issues.

# Set these variables to change default behavior
MOTION_ESTIMATION_METHOD="vibe"  # options: vibe | hmmr
DATASET="ntu"  # options ntu | uestc
BLENDER_CUSTOM_COPY=false

mkdir surreact_project
cd surreact_project
git clone git@github.com:gulvarol/surreact.git

### Step 1: Blender
if [ "$BLENDER_CUSTOM_COPY" = true ]
then
    echo "=> (1/4) Downloading (244MB) and extracting the custom copy of Blender2.92 with dependencies installed"
    echo "=> (1/4) If it fails for some reason, try setting BLENDER_CUSTOM_COPY=false"
    wget https://lsh.paris.inria.fr/surreact/blender-2.92.0-linux64_custom.tar.xz
    tar -xf blender-2.92.0-linux64_custom.tar.xz
    rm blender-2.92.0-linux64_custom.tar.xz
else
    # Download Blender for Linux
    echo "=> (1/4) Downloading and extracting the official Blender2.92 release"
    # Or manually from: https://www.blender.org/download/Blender2.92/blender-2.92.0-linux64.tar.xz
    wget https://download.blender.org/release/Blender2.92/blender-2.92.0-linux64.tar.xz
    echo "=> (1/4) Extracting Blender files, this might take a few minutes..."
    tar -xf blender-2.92.0-linux64.tar.xz
    # Download get-pip.py
    echo "=> (1/4) Installing pip and scipy into Blender's bundled python"
    wget https://bootstrap.pypa.io/get-pip.py
    # Install pip into Blender's bundled python
    blender-2.92.0-linux64/2.92/python/bin/python3.7m get-pip.py
    # Install scipy into Blender's bundled python
    blender-2.92.0-linux64/2.92/python/bin/pip install scipy
    blender-2.92.0-linux64/2.92/python/bin/pip install joblib

    echo "=> (1/4) Creating a new conda environment where additional dependencies will be installed"
    conda create --name extra_blender_env python=3.7
    # To be able to run conda activate:
    source ${CONDA_PREFIX}/etc/profile.d/conda.sh
    # ${CONDA_PREFIX} modified at this point to envs/extra_blender_env
    conda activate extra_blender_env
    # These will be used by the datageneration script
    conda install -c conda-forge openexr-python
    conda install -c conda-forge ffmpeg # x264
    # Hack to copy some libraries into Blender's python
    echo -n "=> (1/4) Confirm this is the path to the extra_blender_env python environment: ${CONDA_PREFIX}"
    read -p "Press enter to continue"
    # Copy OpenEXR packages into Blender's python lib, otherwise would need to add them to ${PYTHONPATH}
    cp -r ${CONDA_PREFIX}/lib/python3.7/site-packages/Imath.py blender-2.92.0-linux64/2.92/python/lib/python3.7/site-packages/
    cp -r ${CONDA_PREFIX}/lib/python3.7/site-packages/OpenEXR* blender-2.92.0-linux64/2.92/python/lib/python3.7/site-packages/
    # Copy the .so files into Blender's lib, otherwise would need to add them to ${LD_LIBRARY_PATH}
    cp ${CONDA_PREFIX}/lib/*.so* blender-2.92.0-linux64/2.92/python/lib/
    # Copy the ffmpeg binary into Blender's bin and add it to ${PATH}, could also add the original location instead
    cp ${CONDA_PREFIX}/bin/ff* blender-2.92.0-linux64/2.92/python/bin

    # The next two are to be able to run extract_J_regressors.py and extract_shape_params.py (not for Blender)
    conda install scipy
    pip install chumpy
fi

### Step 2: smpl_data from SURREAL
echo "=> (2/4) Manual step: Register to SURREAL to obtain <username> <password> https://www.di.ens.fr/willow/research/surreal/data/requestaccess.php"
read -p "Press enter when ready"
echo -n Surreal username: 
read -s USERNAME
echo -n Surreal password: 
read -s PASSWORD
git clone git@github.com:gulvarol/surreal.git
cd surreal/download/
echo "=> (2/4) Downloading smpl_data from surreal, this might take a few minutes..."
./download_smpl_data.sh ../../surreact/datageneration/smpl_data/ ${USERNAME} ${PASSWORD}
# Rearrange file locations
cd ../../surreact/datageneration/smpl_data
mv SURREAL/smpl_data/* .
rm -r SURREAL
rm female_beta_stds.npy
rm male_beta_stds.npy
# Download segm_per_v_overlap.pkl
wget https://raw.githubusercontent.com/gulvarol/surreal/master/datageneration/pkl/segm_per_v_overlap.pkl

### Step 3: smpl_data from SMPL
echo "=> (3/4) Manual step: register to SMPL https://smpl.is.tue.mpg.de/register and download the two files (SMPL_maya.zip, SMPL_python_v.1.0.0.zip) under surreact/datageneration/smpl_data/"
read -p "Press enter when ready"
# Unzip/extract the files
unzip SMPL_maya.zip
unzip SMPL_python_v.1.0.0.zip
# Remove the zip files
rm SMPL_python_v.1.0.0.zip
rm SMPL_maya.zip
rm -r __MACOSX
# Move the two fbx files under smpl_data/
mv SMPL_maya/*.fbx .
rm -r SMPL_maya
cd ../misc/prepare_smpl_data/
echo "=> (3/4) Creating joint_regressors.pkl given the smpl/models/*.pkl"
python extract_J_regressors.py
echo "=> (3/4) Creating shape_params.npz file given the smpl_data.npz"
python extract_shape_params.py
# Cleanup unnecessary files no longer needed
cd ../../smpl_data
rm -r smpl
rm smpl_data.npz

### Step 4: vibe/ and backgrounds/ data
# Back to the root of surreact
echo "=> (4/4) Creating dirs data/ntu and data/uestc"
cd ../../
mkdir -p data/ntu
mkdir -p data/uestc

cd data/${DATASET}

download_and_extract_file () {
echo "=> (4/4) Downloading ${1}, this might take a few minutes..."
wget https://lsh.paris.inria.fr/surreact/${1}
echo "=> (4/4) Extracting ${1}, this might take a few minutes..."
tar -xf ${1}
rm ${1}
}

# Download VIBE/HMMR estimates from NTU/UESTC dataset
download_and_extract_file "${MOTION_ESTIMATION_METHOD}_${DATASET}.tar.gz"
# Download NTU/UESTC background images
download_and_extract_file "backgrounds_${DATASET}.tar.gz"

# Uncomment to download vibe for uestc
# DATASET="uestc"
# download_and_extract_file "${MOTION_ESTIMATION_METHOD}_${DATASET}.tar.gz"
# download_and_extract_file "backgrounds_${DATASET}.tar.gz"

echo "=> Success!"
