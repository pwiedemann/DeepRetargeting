import numpy as np
import cv2 
import matplotlib.pyplot as plt

# Load data
P = np.load("/Users/pablowiedemann/DISTRO/Dev/test_dataset/posmaps.npz")['clips']

# Pick one image
X=P[0,0]

# Image Rotation around "center"
M = cv2.getRotationMatrix2D(center=(256/2,256/2),angle=90,scale=1)
Xrot = cv2.warpAffine(X,M,(256,256))

# Image flip
Xh = cv2.flip( X, 0 ) # horizontal flip
Xv = cv2.flip( X, 1 ) # vertical flip

# Plot images
f, axarr = plt.subplots(2, 2)
axarr[0,0].imshow(X, cmap = 'gray', interpolation = 'bicubic')
axarr[0,1].imshow(Xrot, cmap = 'gray', interpolation = 'bicubic')
axarr[1,0].imshow(Xh, cmap = 'gray', interpolation = 'bicubic')
axarr[1,1].imshow(Xv, cmap = 'gray', interpolation = 'bicubic')


plt.show()
