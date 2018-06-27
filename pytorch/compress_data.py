import numpy as np
import cv2
import os


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
# Returns all images within a folder, as numpy array.
#       * Shape: (f , s)
#           f: frame number
#           s: image size
# =============================================================================
def get_images(folder):
    # list of images in the animation
    l_images=[]
    # get list of all files/(images) contained in "folder" 
    l_filenames = sorted([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
    
    # Iterate over all images and append to image list 
    for f in l_filenames:
        img = cv2.imread(folder+f, cv2.IMREAD_UNCHANGED) 
        l_images.append(img)
    
    return np.array(l_images)


# =============================================================================
# Get Position-map data as numpy array
#   * Shape = (l,f, r, r, c)
#       l: nb. light configurations
#       f: nb. frames
#       r: img. resolution
#       c: nb. channels 
# =============================================================================
def get_posMaps(folder):  
    return get_images(folder)[np.newaxis,...]


# =============================================================================
# Get Light-map data as numpy array
#   * Shape = (l,f, r, r, c)
#       l: nb. light configurations
#       f: nb. frames
#       r: img. resolution
#       c: nb. channels 
# =============================================================================
def get_lightMaps(folder):
    
    l_lightings=[]

    # Get list of all lighting conf. directories 
    l_dirs=sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])
    i=0
    for l in l_dirs:
        subfolder = folder+l+"/"
        anim_images=get_images(subfolder)
        l_lightings.append(anim_images)
        print(i)
        i+=1
    
    return np.array(l_lightings)



# =============================================================================
# compress data
# =============================================================================
def compress():

    # pos_data_dir='/Users/pablowiedemann/DISTRO/Dev/data/position_maps/'   # position maps
    # light_data_dir= '/Users/pablowiedemann/DISTRO/Dev/data/light_maps/'   # light maps
    
    # TEST DATASET
    pos_data_dir='/Users/pablowiedemann/Dev/DATA/toy_dataset/position_maps/'   # position maps
    light_data_dir= '/Users/pablowiedemann/Dev/DATA/toy_dataset/light_maps/'   # light maps

    print("\nLoading posmap data...")
    pos_maps=get_posMaps(pos_data_dir)

    print("Saving pos-maps...")
    np.savez_compressed(pos_data_dir+'posmaps', clips=pos_maps)
    print("Saving pos-maps normalized...")
    pos_maps = standardize(pos_maps)
    np.savez_compressed(pos_data_dir+'posmaps_normalized', clips=pos_maps)

    print("\nLoading lightmap data...")
    light_maps=get_lightMaps(light_data_dir)
    print(light_maps.shape)
    print("Saving light-maps...")
    np.savez_compressed( light_data_dir+'lightmaps', clips=light_maps)
    print("Saving light-maps normalized...")
    light_maps = standardize(light_maps)
    np.savez_compressed(light_data_dir+'lightmaps_normalized', clips=light_maps)




