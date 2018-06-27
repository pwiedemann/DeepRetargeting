"""
My Dataset Class

Pablo Wiedemann  27.06.2018
"""

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
