## DAVID WONG (ANDREW ID DBWONG)
## COLLABORATORS HUSAM WADI, NEERAJ BASU
import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
import util
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from opts import get_opts
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer



def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    filter_scales = opts.filter_scales
    filter_scales = opts.filter_scales
    
    F = len(filter_scales) * 4 # define filter bank size, F = length of scales x 4 types of filters. note filter scales are set in opts; see filter bank for types of filters
    


    # check input img to make sure it is float with range [0,1], else convert it
    if(type(img)==int):
        img = img.astype(np.float32) / 255.0
         
    # duplicate greyscale images into 3 channels then output result as 3F channel img
    if img.shape[0] ==1: # greyscale case
        img = np.repeat(img,3,axis=2)
    
    #if img.shape[2] >3: # 
        #img = img[:,:,0:3]
     
     
    H = img.shape[0]
    W = img.shape[1]
    C = img.shape[2] 
    
    #convert RGB image to Lab color
    img = skimage.color.rgb2lab(img)

    
    #initialize new array [filter_responses] of m X n X 3F with all zeros
    filter_response = np.zeros((H, W, 3*F))        
   
    # filter bank to filter_ouput for fn. return - apply 4 filters and at least 3 scales (these are defined in opts)
    for i in range(C):
        for j in range(len(filter_scales)):

            gaussian = scipy.ndimage.gaussian_filter(img[:,:,i], sigma = filter_scales[j]) # Gaussian
            laplacian_gaussian = scipy.ndimage.gaussian_laplace(img[:,:,i], sigma = filter_scales[j]) # Lapacian of Gaussian
            gaussian_deriv_X = scipy.ndimage.gaussian_filter(img[:,:,i], sigma = filter_scales[j], order = [0,1]) # Gaussian Derivative in X-dir 
            gaussian_deriv_Y = scipy.ndimage.gaussian_filter(img[:,:,i], sigma = filter_scales[j], order = [1,0]) # Gaussian Derivative in Y-dir

            filter_response[:,:,i+3*(4*j+0)] = gaussian
            filter_response[:,:,i+3*(4*j+1)] = laplacian_gaussian
            filter_response[:,:,i+3*(4*j+2)] = gaussian_deriv_X
            filter_response[:,:,i+3*(4*j+3)] = gaussian_deriv_Y

    return filter_response

def compute_dictionary_one_image(args):
        '''
        Extracts a random subset of filter responses of an image and save it to disk
        This is a worker function called by compute_dictionary
        '''
        # ----- TODO -----
        
        i, alpha, image_path = args
        opts = get_opts()
        img = Image.open(os.path.join(opts.data_dir, image_path))
        img = np.array(img).astype(np.float32)/255 
                
        #commence filter response and random sampling of alpha*T pixels, where T is num of training images N
        filter_response = extract_filter_responses(opts,img)
       
        H = filter_response.shape[0]
        W = filter_response.shape[1]
        C = 3 ##filter_response.shape[2]
        filter_response_reshaped = filter_response.reshape((H*W),C)
        x = np.random.choice(filter_response_reshaped[0], alpha, replace=True)
        y = np.random.choice(filter_response_reshaped[1], alpha, replace=True)
        random_sampled_pixels = filter_response_reshaped[x,y,:]

        #sel = np.random.choice(H*W, alpha, replace = False) - old draft for ref
        #sampled_response = filter_response_reshaped[sel] - old draft for ref
        
        np.save(opts.feat_dir, "temp"+"str(%d)"+".npy" %i, random_sampled_pixels)
        print('compute_dictionary_one_img completed')
        return 

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''
    pool = multiprocessing.Pool(processes=n_worker)
    data_dir = opts.data_dir
    #feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    alpha = opts.alpha
    K = opts.K
    filter_response = []
    sampled_response = []
    #args = [] - for multiprocessing

    # load training data and iterate through paths to image files to read images
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    N = len(train_files)   

    #load 
    i=0
    print('loading images')
    for i in range (0,N): 
        img_path = train_files[i]
        print(i)
        #args.append((i, alpha, image_path)) #must use double bracket or args.append screws up! 19Sep20
        img = Image.open(os.path.join(opts.data_dir, img_path)) ####
        img = np.array(img).astype(np.float32)/255 #### 
        filter_response = extract_filter_responses(opts,img)
        H,W,C = filter_response.shape
        filter_response_reshaped = filter_response.reshape((H*W),C) ####
        sel = np.random.choice(H*W, alpha, replace = False) 
        sampled_response = filter_response_reshaped[sel] 
        
        #x = np.random.choice(filter_response_reshaped[0], alpha, replace=True) ####
        #y = np.random.choice(filter_response_reshaped[1], alpha, replace=True) ####
        #random_sampled_pixels = np.concatenate(filter_response_reshaped[x,y,:]) ####
        #all_features = np.append(random_sampled_pixels[i])

    ## call worker fn compute_dict_one_img()         
    #pool.map(compute_dictionary_one_image,args)
    #pool.close()
    #pool.join()
    #print('multiprocessing complete')
    
     

    kmeans = KMeans(n_clusters=K).fit(sampled_response)
    dictionary = kmeans.cluster_centers_
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

    print('compute dictionary () completed')

    return #comput_dict fn



def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    filter_response = extract_filter_responses(opts, img) #call fn to extract filter response
    H, W, channel = np.shape(filter_response) 
    features = filter_response.reshape((H*W, channel)) #reshape filter response matrix into (H*W, C) where C=3

    Euc_distance = scipy.spatial.distance.cdist(features, dictionary, 'euclidean') # measure standard euclidean dist from cluster centre
    wordmap = np.argmin(Euc_distance, axis=1).reshape(H,W) #find min distance and reshape wordmap matrix of same ht and width as img..reshape intointo (H,W)
    return wordmap