import numpy as np


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
