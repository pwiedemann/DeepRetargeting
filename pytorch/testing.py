import numpy as np
import time

import cv2 
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset


# =============================================================================
#  Given two ranges: a = [a_1,a_2]  and   b=[b_1 , b_2]   then 
#  a value "s" in "a", is linearly mapped to a value "t"  in range "b"
# =============================================================================
def mapRange(s,a,b):
    (a1, a2), (b1, b2) = a, b
    t = b1 + ( (s-a1)*(b2-b1) /(a2-a1) )
    return int(t) 




class LightPoseMapDataset(Dataset):

    def __init__(self, light_maps, pose_maps, transform=None):
        self.light_maps = light_maps
        self.pose_maps = pose_maps
        self.transform = transform
    
    # TODO
    def __len__(self):
        return len(self.light_maps)


    def __getitem__(self, idx):
        # Get item 
        lightmap = self.light_maps[idx,...]
        
        # We have a smaller set of pose maps, hence the following index:
        # f.i. if 100 pose maps => light_map 101 corrsp. to pose_map 1, 
        # light_map 102 to pose_map 2 , and so on...
        idx_pose = idx%len(self.pose_maps)
        posemap = self.pose_maps[idx_pose,...] 
        
        sample = {'pose_maps': posemap, 'light_maps': lightmap}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

# =============================================================================
# 
# =============================================================================
def test():
#    X = np.arange(4)[:,np.newaxis]
#    Y = np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6]]) 
    X = np.load("/Users/pablowiedemann/DISTRO/Dev/test_dataset/posmaps.npz")['clips']
    Y = np.load("/Users/pablowiedemann/DISTRO/Dev/test_dataset/lightmaps.npz")['clips']
    
    X = X.reshape(100, 256, 256, 3)
    Y = Y.reshape(300, 256,256,3)
    print(X.shape)
    data = LightPoseMapDataset(light_maps=Y, pose_maps=X)
    dataloader = DataLoader(data, batch_size=4, shuffle=True)
    
    print("dataset:", len(X))
    
    # # Create a autoencoder object
    # model = model.autoencoder()
    # # Define loss function
    # lossFunction = nn.MSELoss()
    # # Define optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-5)
    
    
    num_epochs=1
    for epoch in range(num_epochs):
        for i,img in enumerate(dataloader):
            print("\nbatch: ", i)
            x=img["pose_maps"]
            y=img["light_maps"]
            
            print("X: ", x)
            print("Y: ", y)
           
#         ===================log========================
        print('epoch [{}/{}], loss:{:.4f} ........ time:{} '.format(epoch+1, num_epochs, loss.item(),time.time()-t1))
   
        

def test2():
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F       
    X= torch.randn(1,1,24,24)
    print(X.shape)
   
    pool = nn.MaxPool2d(2, stride=2, return_indices=True)
    unpool = nn.MaxUnpool2d(2,stride=2 )
   
    out,i=pool(X)
    print(i)
   
    print(out.shape)
#
    out2=unpool(out,i)
#
    print(out2.shape)
   
def test3():
    import time
    t1=time.time()
    for i in range(5):
        print('\nepoch [{}/{}] .............. time:{:.4f} \ntrain_loss: {:.4f} \nval_loss: {:.4f}'.format(2, 24,time.time()-t1, 100, 123))
   

test3()