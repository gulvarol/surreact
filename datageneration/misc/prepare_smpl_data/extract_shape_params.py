# From the file provided by SURREAL, extract only the shape parameters data

import numpy as np

smpl_data = np.load("../../smpl_data/smpl_data.npz")
dict_shape = {}
dict_shape["maleshapes"] = smpl_data["maleshapes"][:, :10]
dict_shape["femaleshapes"] = smpl_data["femaleshapes"][:, :10]
np.savez("../../smpl_data/shape_params.npz", **dict_shape)
