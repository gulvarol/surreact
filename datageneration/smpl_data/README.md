### SMPL data

In this folder there should be the following 6 contents:

``` bash
basicModel_f_lbs_10_207_0_v1.0.2.fbx # SMPL female model (SMPL release)
basicModel_m_lbs_10_207_0_v1.0.2.fbx # SMPL male model (SMPL release)
textures/                            # clothing images (also available at lsh.paris.inria.fr/SURREAL/smpl_data/textures.tar.gz) (SURREAL release)
segm_per_v_overlap.pkl               # segmentation of body vertices (SURREAL release)
joint_regressors.pkl                 # vertex-to-joint regressor matrix (SMPL release)
shape_params.npz                     # SMPL shape parameters (SURREAL release)
```

* Download the following two `fbx` files for SMPL for Maya from [https://smpl.is.tue.mpg.de/](https://smpl.is.tue.mpg.de/) using your credentials. Please comply with their license.
  * `basicModel_f_lbs_10_207_0_v1.0.2.fbx`
  * `basicModel_m_lbs_10_207_0_v1.0.2.fbx`
* Download the clothing texture maps from SURREAL, using the [`download_smpl_data.sh`](https://github.com/gulvarol/surreal/blob/master/download/download_smpl_data.sh) script upon accepting the license and receiving credentials. Find more details about this [here](https://github.com/gulvarol/surreal#211-smpl-data).
  * `textures/`
* Download [this file](https://github.com/gulvarol/surreal/blob/master/datageneration/pkl/segm_per_v_overlap.pkl) from SURREAL:
  * `segm_per_v_overlap.pkl`
* Run [`extract_J_regressors.py`](../misc/prepare_smpl_data/extract_J_regressors.py) script to create this file given the SMPL model files `basicModel_(m/f)_lbs_10_207_0_v1.0.0.pkl` (which can also be downloaded from SMPL for Python [https://smpl.is.tue.mpg.de/](https://smpl.is.tue.mpg.de/)).
  * `joint_regressors.pkl`
* Run [`extract_shape_params.py`](../misc/prepare_smpl_data/extract_shape_params.py) script to create this file given the file [`smpl_data.npz` (2.5GB)](https://lsh.paris.inria.fr/SURREAL/smpl_data/smpl_data.npz) from SURREAL.
  *  `shape_params.npz`

*Note: You can remove the following leftover files which are not used by the data generation code:*
``` bash
basicModel_f_lbs_10_207_0_v1.0.0.pkl
basicModel_m_lbs_10_207_0_v1.0.0.pkl
female_beta_stds.npy
male_beta_stds.npy
smpl_data.npz
```