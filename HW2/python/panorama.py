
import numpy as np
import cv2
from opts import get_opts
from planarH import compositeH
import matplotlib.pyplot as plt

## COLLABORATORS: CORINNE ALINI, HUSAM WADI, DANIEL BRONSTEIN, LIU JINKUN, JONATHAN SCHWARTZ, AARUSHI WADHWA

opts = get_opts()

#load images
#imgA = cv2.imread('../data/pano_left.jpg') #- default picture for debugging
#imgB = cv2.imread('../data/pano_right.jpg') 
imgA = cv2.imread('../data/pano_L.jpg') #test image from own camera
imgB = cv2.imread('../data/pano_R.jpg')

img_left = cv2.cvtColor(imgA,cv2.COLOR_BGR2GRAY)
img_right = cv2.cvtColor(imgB,cv2.COLOR_BGR2GRAY)

# find the keypoints and descriptors with ORB - alternative to SIFT
orb = cv2.ORB_create(nfeatures= 5000) #ablation study of top# of matches doesnt yield much visual difference between 1000-5000
locs1, des1 = orb.detectAndCompute(img_left,None)
locs2, des2 = orb.detectAndCompute(img_right,None)

# match features - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
bfmatch = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
matches = bfmatch.match(des1,des2)
sort_matches = sorted(matches, key=lambda x:x.distance)

matched_locs1 = np.array([locs1[match.queryIdx].pt for match in sort_matches[:700]])
matched_locs2 = np.array([locs2[match.trainIdx].pt for match in sort_matches[:700]]) #ablation study of top# of matches doesnt yield much visual difference between 300-700
    
 # compute homography - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html 
H, mask = cv2.findHomography(matched_locs1, matched_locs2, cv2.RANSAC,5.0)

# compose composite image 
composite_pano = compositeH(H, img_left,img_right)
composite_pano = cv2.cvtColor(composite_pano,cv2.COLOR_GRAY2RGB)

plt.imshow(composite_pano)
plt.show()


