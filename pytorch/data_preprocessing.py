import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


# =============================================================================
# Data Normalization
# =============================================================================
def normalize(data, a , b):
    max_array = data.max()*np.ones(shape=data.shape)
    min_array = data.min()*np.ones(shape=data.shape)
    
    return a + (b-a)*(data - min_array) / (max_array - min_array) 

# =============================================================================
# Standardization of input data:
# => Substraction of mean value and division by standard deviation 
# =============================================================================
def standardize(data):
    mean   = data.mean()*np.ones(shape=data.shape)
    std    = data.std()*np.ones(shape=data.shape)
    data = (data - mean)/ std
    return data



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
# Split dataset into training, validation and test sets
#    Rule of thumb: 
#       * test <-- 20% dataset
#       * val  <-- 20% of remaining
#       * training <-- restant data (i.e. if 100 initial data => 64 training data)
#    
#    NOTE: corresp. to current application (only generalizing f.a. lightings), 
#           we now only split the light-map data!
# =============================================================================
def dataset_train_val_test_split(d_dataset, toyset=False):
    
    lMaps = d_dataset['light_maps']
    if toyset:
        lMaps = np.concatenate((lMaps,lMaps,lMaps,lMaps), axis=0)
    
    dim_dataset=len(lMaps)
    idx_test = int(dim_dataset*0.2)
    idx_val = int(dim_dataset*0.8*0.2)
    
    # Test set <-- first 20 animations (from 100)
    test_set = lMaps[:idx_test]
    # Val set <-- last 16 animations (from 100)
    val_set = lMaps[-idx_val:]
    # training 64 inbetween (from 100)
    training_set = lMaps[idx_test:-idx_val]
    
    return {'train_set': training_set, 'val_set': val_set, 'test_set':test_set}
#    return {'train': train_ids, 'val': val_ids, 'test':test_ids }
# =============================================================================
# 
# =============================================================================
def test():
    data_folder="/Users/pablowiedemann/DISTRO/Dev/Data"
    data= load_data(data_folder, toyset=True)
    d_setIds=dataset_train_val_test_split(data, toyset=True)
    for i,j in d_setIds.items():
        print(j.shape)

    
    
# =============================================================================
# Test Class
# =============================================================================
def test2():
    
    from sklearn import preprocessing
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(color_codes=True)
    

    P = np.load("/Users/pablowiedemann/DISTRO/Dev/test_dataset/posmaps.npz")['clips']
    L = np.load("/Users/pablowiedemann/DISTRO/Dev/test_dataset/lightmaps.npz")['clips']
        
    print('max value Light; ', L.max())
    print('min value Light; ', L.min())
    print('max value Pos:' , P.max())
    print('min value Pos:' , P.min())
    
    print('\n------STANDARDIZATION-----------\n')
    L=standardize(L)
    P=standardize(P)
    print('max value Light; ', L.max())
    print('min value Light; ', L.min())
    print('max value Pos:' , P.max())
    print('min value Pos:' , P.min())


#    flat=L.flatten()
#    sns.distplot(flat)

#    print('\n------NORMALIZATION-----------\n')
#    L=normalize(L, -1 , 1)
#    P = normalize(P, -1 , 1)
#    print('max value Light; ', L.max())
#    print('min value Light; ', L.min())
#    print('max value Pos:' , P.max())
#    print('min value Pos:' , P.min())
#    
    plt.clf()
    flat=P.flatten()
    sns.distplot(flat)

    flat=L.flatten()
    sns.distplot(flat)
    
    
test()
    
    
    
    
    
    
    
    