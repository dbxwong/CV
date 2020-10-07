import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts
from briefRotTest import rotTest
from planarH import computeH_norm

## COLLABORATORS: CORINNE ALINI, HUSAM WADI, DANIEL BRONSTEIN, LIU JINKUN, JONATHAN SCHWARTZ

opts = get_opts()
'''
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png') 

matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)

# display matched features
plotMatches(cv_cover, cv_desk, matches, locs1, locs2)

# Q2.1.6 RotTest
rotTest()

#Q2.2.2 for debug
'''
x1 = np.array([[1, 2],
[3,4],
[5,6],
[7,8]])

x2 = np.array([[10,20],
[30,40],
[50,60],
[70,80]])

H = computeH_norm(x1, x2)

x2_test = [[10],[10],[1]]

x1test_output = H*x2_test
x1test_output = x1test_output[:-1]/x1test_output[-1]

print(x1test_output)
