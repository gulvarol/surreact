# Setup for action recognition training

## Download datasets
* [NTU-60 dataset](http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp)
* [UESTC dataset](https://github.com/HRI-UESTC/CFM-HRI-RGB-D-action-database)
* [SURREACT datasets](../README.md#3-download-surreact-datasets)

Download [our preprocessed metadata files - infos.tar.gz (12G)](https://lsh.paris.inria.fr/surreact/infos.tar.gz), to be able to use the training code. Place them so that the structure is:
``` bash
data/
---ntu/
------info/              # provided in infos.tar.gz, metadata
------rgb/avi/           # download from the original dataset the videos (137GB) => (3.6GB resized version)
------*cache.mat         # provided in infos.tar.gz, cached body pose data
---uestc/
------info/              # provided in infos.tar.gz, metadata
------RGBvideo/          # download from the original dataset the videos (82GB)
------mat_from_skeleton/ # download from the original dataset the Kinect joints (5.5GB)
---surreact/             # provided in infos.tar.gz, rest of the structure as explained in README.md
------ntu/hmmr/info/
------ntu/vibe/info/
------uestc/hmmr/info/
------uestc/vibe/info/
```

## Python environment
``` bash
cd training/
# Setup symbolic links to the datasets
mkdir data
ln -s <replace_with_surreact_path> data/surreact
ln -s <replace_with_ntu_path> data/ntu
ln -s <replace_with_uestc_path> data/uestc
# Create surreact_env environment with dependencies
conda env create -f environment.yml
conda activate surreact_env
```

# Run training and testing

We provide sample training and testing launches in the [`exp`](exp) folder for various datasets and configurations used in the paper. You can run them as following: 
```
cd training/
bash exp/surreact_train.sh
```

## Notes:

1) To be able to run flowstreams, you need to download pretrained flow estimation network from and place them under `pretrained_models/` folder:
* [Flow trained on SURREAL and finetuned on SURREACT (52MB)](https://lsh.paris.inria.fr/surreact/flow_surreact_2_v10_fixbg_pretrainsurreal_checkpoint.pth.tar) (default path from [`models/flowstream.py`](models/flowstream.py))
* [Flow trained on SURREAL (52MB)](https://lsh.paris.inria.fr/surreact/flow_surreal_2_S2_checkpoint.pth.tar)

2) The code originally supported more functionalities (estimating segmentation, flow, body pose etc.) which have been removed to simplify. There could be leftovers. The current version is only tested with the action recognition trainings in the `exp/` folder. A few of the functionalities are kept such as using segmentation or flow as input. You are welcome to try these, but the code has not been exhaustively tested and I might not remember the details to provide support.

3) If you see `hri40` in the data/code, this refers to the `uestc` dataset.

4) Filenames have been slightly modified for the UESTC dataset. Check `misc/uestc` folder to see how they were preprocessed.

4) For the experiments in the paper, the video frames for NTU were preprocessed and downsampled to 480x270 pixels resolution to allow faster loading by setting `load_res=0.25`.
