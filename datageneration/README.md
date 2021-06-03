# Setup for data generation

## Option 1: Automatic

We provide a bash script [`setup.sh`](setup.sh) that shows the steps to run to prepare the Blender environment and to download required assets. It is safer to run each line separately than running the whole script at once, because it is intended to provide a guide. You can try running it with `bash setup.sh` and get ready to `Ctrl-C` in case of issues. The steps are explained in comments inside the script as well as below in Option 2.

## Option 2: Manual

### Installation:
Download [Blender](http://download.blender.org/release/). The provided code was tested with [Blender2.92](https://www.blender.org/download/Blender2.92/blender-2.92.0-linux64.tar.xz), with additional packages `pip`, `scipy`, `joblib` installed with [these lines](setup.sh#L33-L38). Create an additional python3.7 environment to install a few packages `openexr-python`, `ffmpeg` with conda as in [these lines](setup.sh#L41-L48). Then either copy the packages under Blender's bundled python or add those paths to environment variables `PYTHONPATH`, `PATH` and `LD_LIBRARY_PATH` as in [these lines](exe/run.sh#L23-L26).

If you want to test our custom Blender package, you can skip the above steps and download it through [here: blender-2.92.0-linux64_custom.tar.xz (244MB)](https://lsh.paris.inria.fr/surreact/blender-2.92.0-linux64_custom.tar.xz).

*Note:* At the time of running the experiments for the paper, and older version [Blender2.78](https://download.blender.org/release/Blender2.78/blender-2.78a-linux-glibc211-x86_64.tar.bz2) was used and an additional python3.5 environment was created to install other dependencies such as `scipy` and `openexr` (due to known problem with `pip` in Blender2.78a). Python3.5 is no longer supported. To make a fresh version, the code has been updated to work with newer Blender versions. If you search the lines with `# blender < 2.8x` in the code, you will notice the modifications required to update from Blender2.78 to Blender2.92. The released datasets were created with the older version.

### Assets:
Several asset files are required to run the synthetic data generation.
1. Please follow the instructions at [`smpl_data/README.md`](smpl_data/README.md) to download these, **make sure to comply with the associated licenses from [SMPL](https://smpl.is.tue.mpg.de/) and [SURREAL](http://www.di.ens.fr/willow/research/surreal/data/license.html)**.
2. In addition, we run motion estimation methods to provide SMPL pose parameters to the data generation. We provide these estimated pose parameters within this project: [vibe_ntu.tar.gz (3.9GB)](https://lsh.paris.inria.fr/surreact/vibe_ntu.tar.gz), [hmmr_ntu.tar.gz (14GB)](https://lsh.paris.inria.fr/surreact/hmmr_ntu.tar.gz), [vibe_uestc.tar.gz (12GB)](https://lsh.paris.inria.fr/surreact/vibe_uestc.tar.gz), [hmmr_uestc.tar.gz subset (420MB)](https://lsh.paris.inria.fr/surreact/hmmr_uestc.tar.gz)]. **Please cite the original datasets [NTU](http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp) and [UESTC](https://github.com/HRI-UESTC/CFM-HRI-RGB-D-action-database/blob/master/License%20Agreement.pdf)** if you use these motion estimation data. See [`misc/motion_estimation`](misc/motion_estimation) to get an idea on how the motion estimates were obtained using VIBE or HMMR. **Please comply with their licenses if you use code from [VIBE](https://github.com/mkocabas/VIBE) or [HMMR](https://github.com/akanazawa/human_dynamics)**. Note that we have removed `verts` data to retain small storage, but the vertices can be obtained given the pose and shape parameters using the SMPL forward function.
3. Finally, we used scripts in [`misc/background_crops`](misc/background_crops) to create background images from target datasets, you can download them here [backgrounds_ntu.tar.gz (225MB)](https://lsh.paris.inria.fr/surreact/backgrounds_ntu.tar.gz) and [backgrounds_uestc.tar.gz (323MB)](https://lsh.paris.inria.fr/surreact/backgrounds_uestc.tar.gz). You can also use LSUN images from SURREAL or any images to start.

# Run data generation
Set the variables `BLENDER_PATH` and `CODE_PATH` in [`exe/run.sh`](exe/run.sh), then run synthetic data generation on a single video clip:
``` bash
bash exe/run.sh
```
