## DAVID WONG (ANDREW ID DBWONG)
from sklearn.preprocesesing import StandardScaler
from sklearn import svm, metrics
import numpy as np
import scipy.stats
from scipy.spatial import distance

scaler = StandardScaler()
def custom:
    #resize images


    # SUBTRACT MEAN COLOR DURING NORMALIZING
    ''' use this to replace all instances of normalizing training images before obtaining wordmap (see example below)
    BEFORE:
        train_image_path = train_files[i]
        train_img = Image.open(os.path.join(opts.data_dir, test_image_path))
        train_img = np.array(img).astype(np.float32)/255 
         
    AFTER(SEE CODE BLOCK BELOW)
    '''
    N = len(test_files)   
    scaler = StandardScaler()
    for i in range(N):
        train_image_path = train_files[i]
        train_img = Image.open(os.path.join(opts.data_dir, test_image_path))
        train_img = np.array(img).astype(np.float32)/255 ####
        train_img = scaler(test_img) 
    continue

    # USE OF ALTERNATIVE DISTANCE FUNCTIONS
    ## 1) HELLINGER DISTANCE

    def HELLINGER:
        
        features = np.asarray([0.03, 0.5, 0.3, 0.2]) #sample histogram data
        dictionary = np.asarray([0.07, 0.03, .6, .2]) #sample histogram data
        features /= features.sum()
        dictionary /= dictionary.sum()
        hell = np,sqrt(np.sum((np.sqrt(p)-np.sqrt(q)**2))) / _sqrt2
        dist_Hel =  hell(features,dictionary)
    
        return dist_Hel

    ## 2) MANHATTAN OR CITY BLOCK DISTANCE

    def MANHATTAN:
        features = np.asarray([0.03, 0.5, 0.3, 0.2]) #sample histogram data
        dictionary = np.asarray([0.07, 0.03, .6, .2]) #sample histogram data
        distance_Manhat = distance.cityblock(features,dictionary)
        
        return distance_Manhat
    
    def CHEBYSHEV:
        features = np.asarray([0.03, 0.5, 0.3, 0.2]) #sample histogram data
        dictionary = np.asarray([0.07, 0.03, .6, .2]) #sample histogram data
        distance_Cheb = distance.cityblock(features,dictionary)

        return distance_Cheb
    
    def SVM:

        ## Using a SVM classfier
        SVM_classifier = svm.SVC(kernel='linear') #use Linear Kernel
    
        ## train the model using training sets
         SVM_classifier.fit(train_img,train_label)

        ## predict response for test dataset
        test_pred =  SVM_classifier.predict(test_img)
        print("Accuracy:", metrics.accuracy_score(test_label, test_pred))
        print("Recall:", metrics.recall_score(test_label,test_pred))

        return test_pred