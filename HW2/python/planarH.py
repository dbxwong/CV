import numpy as np
import cv2


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points

# x1 x2 are coord of 2 different images; x1 = img 1, x2 = img2;
# x1 and x2 are N X 2 matrices


	assert(x1.shape[0]==x2.shape[0]) # ensure x1 and x2 index 0's (N) is of same shape
	assert(x1.shape[1]==2)
	assert(x2.shape[1]==2)

	N = x1.shape[0]
	
	A = np.zeros(2*N, 9)


# Compute homography matrix
	for i in range(N):

		x,y = x1[i][0], x1[i][1]
		u,v = x2[i][0], x2[i][1]

		A[2*i]=[x,y,1,0,0,0,-u*x, -u*y, -u] #append matrix A
		A[2*i+1]=[0,0,0,x,y,1,-v*x, -v*y, -v]

	A = np.asarray(A)
	U,S,V = np.linalg.svd(A) # V columns corresppond to the eigenvectors of A^-1A
	h = V[:,end] # From the SVD we extract the right singular vector from V which corresponds to the smallest singular value
	H2to1 = h.reshape(3,3) # reshape H into a 3x3 matrix
     
	return H2to1


def computeH_norm(x1, x2): # x1 and x2 are N X 2 matrices
	#Q2.2.2
	# x1 and x2 are N X 2 matrices

	n = x1.shape[0]
	if x2.shape[0] !=n:
		print('Error: number of points dont match')

	#Compute the centroid of the points 
	#Shift the origin of the points to the centroid
	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	 x1 = x1 / x1[0]	
	 mean_x1 = np.mean(x1[:1], axis=1)
	 S1 = np.sqrt(2)/np.std(x1[0:])
	 T1 = np.array([[S1 , 0 , -S1*mean_x1[0]], [ 0 , S1, -S1*mean_x1[1]] , [0,0,1]]) #T = [[1/sx, 0, -mx/sx], [0, 1/sy, -my/sy], [0, 0, 1]] normalizing transformation in matrix form
	 x1 = np.dot(T1,x1)

	 x2 = x2 / x2[0]	
	 mean_x2 = np.mean(x2[:1], axis=1)
	 S2 = np.sqrt(2)/np.std(x1[0:])
	 T2 = np.array([[S2 , 0 , -S2*mean_x2[0]], [ 0 , S2, -S2*mean_x2[1]] , [0,0,1]]) #T = [[1/sx, 0, -mx/sx], [0, 1/sy, -my/sy], [0, 0, 1]] normalizing transformation in matrix form
	 x2 = np.dot(T2,x2)

	#Similarity transform 1


	#Similarity transform 2


	#Compute homography
	computeH(x1, x2)

	#Denormalization
	

	return H2to1




def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
	'''
	input: locs1, locs2 are Nx2 matrices of matched points

	output: bestH2to1 - homography H with most inliers found during RANSAC. x2 is a point in locs2; x1 is a corresponding point in locs1. 
			inliers - vector of length N with a 1 at those matches that are part of consensus set, else 0

	'''
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for = default 500
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier = default 2

	if num_inliers > max_num

	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	

	#Create mask of same size as template

	#Warp mask by appropriate homography

	#Warp template by appropriate homography

	#Use mask to combine the warped template and the image
	
	return composite_img

'''
