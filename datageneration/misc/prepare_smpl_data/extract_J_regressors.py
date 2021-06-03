# From the SMPL pkl files provided by the official SMPL release, extract the J_regressor for both male and female models

"""
The joint_regressor used in SURREAL is a subset of the J_regressor from the female model, with the regression_verts indices.
Therefore, the following is True.
dict_regressors = pkl.load(open('smpl_data/joint_regressors.pkl', 'rb'))
np.all(dict_regressors['J_regressor_female'][:, smpl_data['regression_verts']] == smpl_data['joint_regressor'])

The following is false, the two regressors are not the same
(dict_regressors['J_regressor_female'][:] == dict_regressors['J_regressor_male'][:])).nnz == 0

# source activate pytorch1
"""

import pickle as pkl

model_f = pkl.load(
    open("../../smpl_data/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl", "rb"),
    encoding="latin1",
)
# Yes, there is a typo in the release with lowercase m
model_m = pkl.load(
    open("../../smpl_data/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl", "rb"),
    encoding="latin1",
)

dict_regressors = {
    "J_regressor_female": model_f["J_regressor"],
    "J_regressor_male": model_m["J_regressor"],
}

pkl.dump(dict_regressors, open("../../smpl_data/joint_regressors.pkl", "wb"))
