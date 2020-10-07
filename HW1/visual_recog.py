## DAVID WONG (ANDREW ID DBWONG)

import os, math, multiprocessing
from os.path import join
from copy import copy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
import visual_words
import util
from sklearn.metrics import confusion_matrix



def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    K = opts.K
    hist, bin_edges = np.histogram(wordmap.reshape(-1,1), bins = range(K + 1), density = True)
    
    #plt.hist(hist, bins = range(K+1),density=True)  ##for debugging
    #plt.show()  ##for debugging
    return hist
    

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K #dictionary size
    L = opts.L #number of layer in spatial pyramid
       
    hist_all=[]
    H = wordmap.shape[0]
    W = wordmap.shape[1]
    #num_cells = (2**L) #length of cells along edge  remember not to use (2**layernum)^2 as it didnt work
    #h = H // num_cells  # height of subdivisions
    #w = W // num_cells # width of subdivisions
    
    for i in range(L+1):

        for L in range (0, L):
            layer_number = i
            if L == 0 or L == 1:
                weight = (2**(-layer_number)) # weight set as 2^(-L) for layer 0 and 1

            else: 
                weight = (2**(layer_number-L-1)) # weight set as 2^((layer_number)-L-1)) for rest of layers

            num_cells =  (2**L)

            x = np.array_split(wordmap, num_cells, axis=0)
            for rows in x:
                y = np.array_split(rows,num_cells,axis=1)
                for cols in y:
                    hist, bin_edges = np.histogram(cols,bins=K+1, density = True)
                    hist_all = np.append(hist_all, hist/((H*W)*weight))
                   
    hist_all = hist_all/hist_all.sum()
    print('SPM fn completed')                
    return hist_all
    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    out_dir = opts.out_dir
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    img = Image.open(os.path.join(opts.data_dir, img_path)) #load image
    img = np.array(img).astype(np.float32)/255 
    wordmap = visual_words.get_visual_words(opts, img, dictionary) #extract wordmap
    feature = get_feature_from_wordmap_SPM(opts, wordmap) #compute SPM
    
    return feature # return computed feature

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L
    K = opts.K

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    N = len(train_files)
    
    features = []
    labels = []
    features=[]
    N = len(train_files)
    
    for i in range (0,N):
        dictionary = np.load(join(out_dir, 'dictionary.npy'))
        img_path = train_files[i]
        feature = get_image_feature(opts, img_path, dictionary) #returns computed feature
        features = feature[i,:] ## TA's recommendation. 22/9
                
        #compute labels
        folder, file = os.path.split(img_path)
        labels = np.append(labels,compute_labels(folder))
        
    ## multiprocessing attmept
    #pool = multiprocessing.Pool(processes=n_worker)
    # write the above in one custom function

    ## example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num)
    print('build recog system complete')
    return

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    #K = opts.K # num of visual words (dictionary size) currently set at 10
    sim=[]
    minimum = []
    minimum = np.minimum(word_hist,histograms)
    sim = np.sum(min)
    return sim    
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']
    features = trained_system['features']
    labels = trained_system['labels']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    conf = np.zeros((8,8))
    N = len(test_files)

    for i in range(0,N):
        print(i)   
        img_path = test_files[i]
        
        test_feature = get_image_feature(opts, img_path, dictionary)
                
        folder, file = os.path.split(img_path)
        test_labels[i] = compute_labels(folder)
        similarity = distance_to_set(test_feature,features) #note features here is trained features
        predicted_label = labels[np.argmax(similarity)] #note labels here is trained labels
        
        conf[test_labels[i], predicted_label]
                     
    np.save(out_dir,"conf.npy",conf)
    
    accuracy = np.trace(conf)/np.sum(conf)
    print('Accuracy is:', accuracy)
    return conf,accuracy

def compute_labels (folder):
    
    if folder == 'aquarium':
        return(0)
    elif folder == 'desert':
        return(1)
    elif folder == 'highway':
        return(2) 
    elif folder == 'kitchen':
        return(3)
    elif folder == 'laundromat':
        return(4)
    elif folder == 'park':
        return(5)
    elif folder == 'waterfall':
        return(6)
    elif folder == 'windmill':
        return(7)
    
