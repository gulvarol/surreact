# SURREACT: Synthetic Humans for Action Recognition from Unseen Viewpoints

[Gül Varol](https://imagine.enpc.fr/~varolg/), [Ivan Laptev](http://www.di.ens.fr/~laptev/) and [Cordelia Schmid](https://thoth.inrialpes.fr/~schmid/),  [Andrew Zisserman](https://www.robots.ox.ac.uk/~az/),
*Synthetic Humans for Action Recognition from Unseen Viewpoints*, IJCV 2021.

[[Project page]](http://www.di.ens.fr/willow/research/surreact/) [[arXiv]](https://arxiv.org/abs/1912.04070)

<p align="center">
<img src="http://www.di.ens.fr/willow/research/surreact/images/surreact.jpg" height="300">
</p>

## Contents
* [1. Synthetic data generation from motion estimation](#1-synthetic-data-generation-from-motion-estimation)
* [2. Training action recognition models](#2-training-action-recognition-models)
* [3. Download SURREACT datasets](#3-download-surreact-datasets)
* [Citation](#citation)
* [License](#license)
* [Acknowledgements](#acknowledgements)

## 1. Synthetic data generation from motion estimation

Please follow the instructions at [`datageneration/README.md`](datageneration/README.md) for setting up the Blender environment and downloading required assets.

Once ready, you can generate one clip by running:
``` bash
# set `BLENDER_PATH` and `CODE_PATH` variables in this script
bash datageneration/exe/run.sh
```
Note that `-t 1` option in [`run.sh`](datageneration/exe/run.sh) can be removed to run faster on multi cores. We used [`submit_multi_job*.sh`](datageneration/exe/) to generate clips for the whole datasets in parallel on the cluster, you can adapt this for your infrastructure. This script also has sample argument-value pairs. Find in [`utils/argutils.py`](datageneration/utils/argutils.py) a list of arguments and their explanations. You can enable/disable outputting certain modalities by setting [`output_types` here](datageneration/main.py#L72).

## 2. Training action recognition models

Please follow the instructions at [`training/README.md`](training/README.md) for setting up the Pytorch environment and preparing the datasets.

Once ready, you can launch training by running:
``` bash
cd training/
bash exp/surreact_train.sh
```

## 3. Download SURREACT datasets

In order to download SURREACT datasets, you need to accept the license terms from *SURREAL*. The links to license terms and download procedure are available here:

https://www.di.ens.fr/willow/research/surreal/data/

Once you receive the credentials to download the dataset, you will have a personal username and password. Use these to download the synthetic videos from the following links. Note that due to storage complexity, we only provide `.mp4` video files and metadata, but not the other modalities such as flow and segmentation. You are encouraged to run the data generation code to obtain those. We provide videos corresponding to NTU and UESTC datasets.

* [surreact_ntu_vibe.tar.gz, (8.4GB)](https://lsh.paris.inria.fr/SURREAL/surreact/surreact_ntu_vibe.tar.gz) with 105,642 videos (105,162 training, 480 test). This is used in Table 1 of the paper, obtains the best results.
* [surreact_ntu_hmmr.tar.gz, (9.1GB)](https://lsh.paris.inria.fr/SURREAL/surreact/surreact_ntu_hmmr.tar.gz) with 105,983 videos (105,503 training, 480 test). This is used in most experiments in the paper.
* [surreact_uestc_vibe.tar.gz, (3.2GB)](https://lsh.paris.inria.fr/SURREAL/surreact/surreact_uestc_vibe.tar.gz) with 12800 videos (12800 training, 0 test). This is not used in the paper.
* [surreact_uestc_hmmr.tar.gz, (646MB)](https://lsh.paris.inria.fr/SURREAL/surreact/surreact_uestc_hmmr.tar.gz) with 3193 videos (3154 training, 39 test). This is a subset due to computational complexity, it is used in the paper.

The structure of the folders can be as follows:

``` shell
surreact/
------- uestc/  # using motion estimates from the UESTC dataset
------------ hmmr/
------------ vibe/
------- ntu/  # using motion estimates from the NTU dataset
------------ hmmr/
------------ vibe/
---------------- train/
---------------- test/
--------------------- <sequenceName>/ # e.g. S001C002P003R002A001 for NTU, a25_d1_p048_c1_color.avi for UESTC
------------------------------ <sequenceName>_v%03d_r%02d.mp4       # RGB - 240x320 resolution video
------------------------------ <sequenceName>_v%03d_r%02d_info.mat  # metadata
# bg         [char]          - name of the background image file
# cam_dist   [1 single]      - camera distance
# cam_height [1 single]      - camera height
# cloth      [chat]          - name of the texture image file
# gender     [1 uint8]       - gender (0: 'female', 1: 'male')
# joints2D   [2x24xT single] - 2D coordinates of 24 SMPL body joints on the image pixels
# joints3D   [3x24xT single] - 3D coordinates of 24 SMPL body joints in world meters
# light      [9 single]      - spherical harmonics lighting coefficients
# pose       [72xT single]   - SMPL parameters (axis-angle)
# sequence   [char]          - <sequenceName>
# shape      [10 single]     - body shape parameters
# source     [char]          - 'ntu' | 'hri40'
# zrot_euler [1 single]      - rotation in Z (euler angle), zero

# *** v%03d stands for the viewpoint in euler angles, we render 8 views: 000, 045, 090, 135, 180, 225, 270, 315.
# *** r%02d stands for the repetition, when the same video is rendered multiple times (this is always 00 for the released files)
# *** T is the number of frames, note that this can be smaller than the real source video length due to motion estimation dropping frames
```

## Citation
If you use this code or data, please cite the following:

```
@INPROCEEDINGS{varol21_surreact,  
  title     = {Synthetic Humans for Action Recognition from Unseen Viewpoints},  
  author    = {Varol, G{\"u}l and Laptev, Ivan and Schmid, Cordelia and Zisserman, Andrew},  
  booktitle = {IJCV},  
  year      = {2021}  
}
```

## License
Please check the [SURREAL license terms](http://www.di.ens.fr/willow/research/surreal/data/license.html) before downloading and/or using the SURREACT data and data generation code.

## Acknowledgements
The data generation code was extended from [gulvarol/surreal](https://github.com/gulvarol/surreal). The training code was extended from [bearpaw/pytorch-pose](https://github.com/bearpaw/pytorch-pose). The source of assets include action recognition datasets [NTU](http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp) and [UESTC](https://github.com/HRI-UESTC/CFM-HRI-RGB-D-action-database/blob/master/License%20Agreement.pdf), [SMPL](https://smpl.is.tue.mpg.de/) and [SURREAL](http://www.di.ens.fr/willow/research/surreal/) projects. The motion estimation was possible thanks to [mkocabas/VIBE](https://github.com/mkocabas/VIBE) or [akanazawa/human_dynamics (HMMR)](https://github.com/akanazawa/human_dynamics) repositories. Please cite the respective papers if you use these.

Special thanks to Inria clusters `sequoia` and `rioc`.
