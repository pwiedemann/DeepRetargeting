"""
AFFINE IMAGE TRANSORMS (Rotations and Flips) for DATA AUGMENTATION

   Input: img of size (w,h,c)
   w: image width
   h: image height
   c: image channels

 
Pablo Wiedemann  27.06.2018 
""" 

import numpy as np
import cv2 
import matplotlib.pyplot as plt

# =============================================================================
#  Rotate image around center with input angle (default in degrees)
# =============================================================================
def rotate(img, angle , radians=False):
    
    w = img.shape[0]
    h = img.shape[1]
    
    # Image Rotation around "center"
    M = cv2.getRotationMatrix2D(center=(w/2,h/2),angle=angle,scale=1)
    return cv2.warpAffine(img,M,(256,256))
    

# =============================================================================
# Flip image around axis. 
#   If axis = 0 => horizontal flip 
#   If axis = 1 => vertical flip 
# =============================================================================
def flip(img, axis):
    assert axis ==0 or axis ==1
    # Image flip
    return cv2.flip( img, axis ) 
   


# =============================================================================
# PLot examples
# =============================================================================
def show_example_trafos(img):
    f, axarr = plt.subplots(2, 2)
    axarr[0,0].imshow( img,             cmap = 'gray', interpolation = 'bicubic')
    axarr[0,0].set_title('original')
    axarr[0,0].set_xticklabels([])
    axarr[0,0].set_yticklabels([])
    
    axarr[0,1].imshow(rotate(img, 90),  cmap = 'gray', interpolation = 'bicubic')
    axarr[0,1].set_title('rot by 90 degrees')
    axarr[0,1].set_xticklabels([])
    axarr[0,1].set_yticklabels([])
    
    axarr[1,0].imshow(flip(img,0),      cmap = 'gray', interpolation = 'bicubic')
    axarr[1,0].set_title('horizontal flip')
    axarr[1,0].set_xticklabels([])
    axarr[1,0].set_yticklabels([])
    
    axarr[1,1].imshow(flip(img,1),      cmap = 'gray', interpolation = 'bicubic')
    axarr[1,1].set_title('vertical flip')
    axarr[1,1].set_xticklabels([])
    axarr[1,1].set_yticklabels([])
    
    plt.show()



#X = np.load("/Users/pablowiedemann/Dev/DATA/toy_dataset/posmaps.npz")['clips']
#img=X[0][0]
#show_example_trafos(img)