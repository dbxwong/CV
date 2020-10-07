import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from planarH import computeH_ransac
import matplotlib.pyplot as plt
from matchPics import matchPics

## COLLABORATORS: CORINNE ALINI, HUSAM WADI, DANIEL BRONSTEIN, LIU JINKUN, JONATHAN SCHWARTZ, AARUSHI WADHWA

opts = get_opts()

# Read images
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png') 
hp_cover =  cv2.imread('../data/hp_cover.jpg')
hp_cover_resized = cv2.resize(hp_cover,(cv_cover.shape[1],cv_cover.shape[0])) #resized so it would overlay on cv_cover better Q2.2.4


# compute homography of cv_cover and cv_desk automatically using MatchPics and computeH_ransac
matches, locs1, locs2 = matchPics(cv_desk, cv_cover, opts)

matched_locs1 = np.array([locs1[i] for i in matches[:0]])
matched_locs2 = np.array([locs2[i] for i in matches[:1]]) 
bestH2to1, inliers = computeH_ransac(matched_locs1, matched_locs2, opts)
#H, mask = cv2.findHomography(matched_locs1, matched_locs2, cv2.RANSAC,5.0) ## debugging to test the rest of non planarH code

compositeImg = compositeH(bestH2to1, hp_cover_resized, cv_desk)
plt.imshow(compositeImg)

compositeImg_RGB = cv2.cvtColor(compositeImg, cv2.COLOR_BGR2RGB) 
plt.imshow(compositeImg_RGB)

