import numpy as np
import torch

# =============================================================================
#  DATA LOADER
#    default: returns the entire normalized (actually standardized) dataset 
# =============================================================================
def load_data(folder, toyset=False, standardized=True):
    if not folder.endswith("/"):
        folder+="/"
    print(folder)
    if toyset:
        p = np.load(folder+"toy_dataset/posmaps.npz")['clips']
        l = np.load(folder+"toy_dataset/lightmaps.npz")['clips']
    elif standardized:
        p = np.load(folder+"dataset/posmaps_normalized.npz")['clips']
        l = np.load(folder+"dataset/lightmaps_normalized.npz")['clips']
    else:
        p = np.load(folder+"dataset/posmaps.npz")['clips']
        l = np.load(folder+"dataset/lightmaps.npz")['clips']
    
    return {'light_maps': l , 'pose_maps': p}


# =============================================================================
# Reshape data to fit the input of the networks
# and convert to pytorch tensor
# =============================================================================
def reshape_data(X):
    Xshape=X.shape
    # Flatten first 2 dimensions:
    X= X.reshape(Xshape[0]*Xshape[1],Xshape[2],Xshape[3],Xshape[4])
    # Move channel dimension to second array axis (channels first)
    X=np.moveaxis(X,-1,1)
    # Convert to tensor
    X=torch.from_numpy(X).double()
    return X
