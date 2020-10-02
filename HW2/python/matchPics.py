import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection


def matchPics(I1, I2, opts):
	#I1, I2 : Images to match
	#opts: input opts
	ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
	sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
	locs1 = [] # N x 2 matrix containing x,y coords or matched point pairs
	locs2 = [] # N x 2 matrix containing x,y coords or matched point pairs
	matches = [] # p x 2 matrix - first col are indices for descriptor of I1 features; 2nd col are indices related to I2

	#Convert Images to GrayScale
	img1_grey = cv2.cvtColor(I1,cv2.COLOR_BGR2GRAY)
	img2_grey = cv2.cvtColor(I2,cv2.COLOR_BGR2GRAY)

	#Detect Features in Both Images with corner_detection() fn.
	locs1 = corner_detection(img1_grey, sigma)
	locs2 = corner_detection(img2_grey, sigma)

	#Obtain descriptors for the computed feature locations with computeBrief() fn.
	img1_descrp, locs1 = computeBrief(img1_grey, locs1)
	img2_descrp, locs2 = computeBrief(img2_grey, locs2)

	#Match features using the descriptors
	matches = briefMatch(img1_descrp, img2_descrp, ratio)
	
	##For debugging
	#print (matches.shape)
	#print (len(matches))
	#print (locs1.shape)
	#print (locs2.shape)

	return matches, locs1, locs2
