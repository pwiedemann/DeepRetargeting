import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import time

from my_utils import load_data
import model

# =============================================================================
# My dataset class
# =============================================================================
class LightPoseMapDataset(Dataset):

    def __init__(self, light_maps, pose_maps, transform=None):
        self.light_maps = light_maps
        self.pose_maps = pose_maps
        self.transform = transform
    
    def __len__(self):
        return len(self.light_maps)


    def __getitem__(self, idx):
        # Get current light map item 
        lightmap = self.light_maps[idx,...]
        
        # Get initial lightmap of current lighting configuration
        idx_initLM = idx - idx%len(self.pose_maps)
        init_lightmap = self.light_maps[idx_initLM,...]
        # Get current position map:
        #   We have a smaller set of pose maps, hence the following index:
        #   f.i. if 100 pose maps => light_map 101 corrsp. to pose_map 1, 
        #   light_map 102 to pose_map 2 , and so on...
        idx_pose = idx%len(self.pose_maps)
        posemap = self.pose_maps[idx_pose,...] 
        print('samples: ', [idx , idx_initLM, idx_pose])
        
        sample = {'pose_maps': posemap, 'light_maps': lightmap, 'init_light_map': init_lightmap}
        
        if self.transform:
            sample = self.transform(sample)        
        
        return sample



# =============================================================================
# Training function
# =============================================================================
def train(training_data, validation_data=None):
    
    training_dataset = LightPoseMapDataset(light_maps=training_data['light_maps'], 
                                           pose_maps=training_data['pose_maps'])
    
    train_dataloader = DataLoader(training_dataset, batch_size=len(training_dataset), shuffle=True)
    
    print('trainingset: ', len(training_dataset))
    
    if validation_data:
        val_dataset = LightPoseMapDataset(light_maps=validation_data['light_maps'], 
                                               pose_maps=validation_data['pose_maps'])
        
        val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)
    
    
    
    # model objects
    encoder     = model.Encoder()
    decoder     = model.Decoder()
    retargeter  = model.Decoder()
    encoder.double()
    decoder.double()
    retargeter.double()
    
    
    # Define loss function
    mse_loss = nn.MSELoss()
        
    # Define optimizer
    params = list(encoder.parameters()) + list(decoder.parameters()) + list(retargeter.parameters())
    optimizer = torch.optim.Adam(params, lr=0.01, weight_decay=1e-5)
    
    num_epochs=1
    for epoch in range(num_epochs):
        print("\nepoch: ", epoch+1)
        t1=time.time()
        for i, data in enumerate(train_dataloader):
            print("in trianing batch: ", i)
            light_map = data["light_maps"]      # current light map
            pose_map = data["pose_maps"]        # current pose map
            init_LM = data['init_light_map']    # initial light map (to retarget) 
            
            in_AE = torch.cat((light_map,pose_map),dim=1)
            in_Retarget = torch.cat((init_LM,pose_map),dim=1)
            
            # ===================forward=====================
            # forward pass "Autoencoder Net"
            out_AE = encoder(in_AE)
            out_AE = decoder(out_AE)
            # forward pass "Retargeter Net"
            out_Retarget = encoder(in_Retarget)
            out_Retarget = retargeter(out_Retarget)
            
            # Set losses
            loss_AE = mse_loss(out_AE, light_map)
            loss_Retarget = mse_loss(out_Retarget, light_map)
            
            training_loss=loss_AE+loss_Retarget
            
            # ===================backward====================
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()
        
        if validation_data:
            for i, data in enumerate(val_dataloader):
                print("In validation batch: ", i)
                light_map = data["light_maps"]      # current light map
                pose_map = data["pose_maps"]        # current pose map
                init_LM = data['init_light_map']    # initial light map (to retarget) 
                
                in_AE = torch.cat((light_map,pose_map),dim=1)
                in_Retarget = torch.cat((init_LM,pose_map),dim=1)
                
                # ===================forward=====================
                # forward pass "Autoencoder Net"
                out_AE = encoder(in_AE)
                out_AE = decoder(out_AE)
                # forward pass "Retargeter Net"
                out_Retarget = encoder(in_Retarget)
                out_Retarget = retargeter(out_Retarget)
                
                # Set losses
                loss_AE = mse_loss(out_AE, light_map)
                loss_Retarget = mse_loss(out_Retarget, light_map)
                
                val_loss=loss_AE+loss_Retarget
                    
        # ===================log========================
        print('\nepoch [{}/{}] .............. time:{:.4f} \ntrain_loss: {:.4f} \nval_loss: {:.4f}'.format(2, 24,time.time()-t1, training_loss, val_loss))
   
    torch.save(encoder.state_dict(),"/Users/pablowiedemann/DISTRO/Dev/test_dataset/encoder.pt")
    torch.save(decoder.state_dict(), "/Users/pablowiedemann/DISTRO/Dev/test_dataset/decoder.pt")



# =============================================================================
# 
# =============================================================================
def prepare_rawdata(X):
    Xshape=X.shape
    # Flatten first 2 dimensions:
    X= X.reshape(Xshape[0]*Xshape[1],Xshape[2],Xshape[3],Xshape[4])
    # Move channel dimension to second array axis (channels first)
    X=np.moveaxis(X,-1,1)
    # Convert to tensor
    X=torch.from_numpy(X).double()
    return X

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
def dataset_train_val_test_split(dataset, toyset=False):
    
    if toyset:
        dataset = np.concatenate((dataset,dataset,dataset,dataset), axis=0)
    
    dim_dataset=len(dataset)
    idx_test = int(dim_dataset*0.2)
    idx_val = int(dim_dataset*0.8*0.2)
    
    # Test set <-- first 20 animations (from 100)
    test_set = dataset[:idx_test]
    # Val set <-- last 16 animations (from 100)
    val_set = dataset[-idx_val:]
    # training 64 inbetween (from 100)
    training_set = dataset[idx_test:dim_dataset-idx_val]
    return {'train_set': training_set, 'val_set': val_set, 'test_set':test_set}
       

# =============================================================================
# MAIN
# =============================================================================
def main():
    
    # DATA
    data_folder = data_folder="/Users/pablowiedemann/DISTRO/Dev/Data/"
    data = load_data(data_folder, toyset=True)
    
    data['pose_maps'] = data['pose_maps'][:,0:1,...]
    data['light_maps'] = data['light_maps'][:,0:1,...]
    
    data['pose_maps'] = prepare_rawdata(data['pose_maps'])
    
    # Split data 
    d_set = dataset_train_val_test_split(data['light_maps'], toyset=True)
    training_data  = {
            'light_maps'    : prepare_rawdata(d_set['train_set']) , 
            'pose_maps'     : data['pose_maps']
            }
    val_data = {
            'light_maps'    : prepare_rawdata(d_set['val_set']) , 
            'pose_maps'     : data['pose_maps']
            }
    test_data = {
            'light_maps'    : prepare_rawdata(d_set['test_set']) , 
            'pose_maps'     : data['pose_maps']
            }
    
    # TRAINING 
#    train(training_data, validation_data=val_data)

main()