from train_test_split import split
from hog_feature_extractor import HOGFeatureExtractor
from sift_feature_extractor import sift
from datetime import datetime
#import cv2
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import *

def load_feature():
    """
    load features directly
    """
    
    with open('features/hog_train.dat', 'rb') as f:
        hog_train = pickle.load(f)
    with open('features/hog_test.dat', 'rb') as f:
        hog_test = pickle.load(f)
    with open('features/sift_train.dat', 'rb') as f:
        sift_train = pickle.load(f)
    with open('features/sift_test.dat', 'rb') as f:
        sift_test = pickle.load(f)
    return hog_train, hog_test, sift_train, sift_test


def feature_extractor(X_train, X_test):
    
    """
    return hog and sift features for both train and test set
    """
    
    hog_train = []
    hog_test = []
    sift_train = []
    sift_test = []
    hog = cv2.HOGDescriptor()
    #HOGFeatureExtractor()
    
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    winStride = (8,8)
    padding = (8,8)
    locations = ((10,20),)
    
    for img in X_train:
        kps, descs = sift(img)
        #if len(img.shape) == 2 :
            #img = img[:,:,numpy.newaxis]
        hog_train.append(hog.compute(img,winStride,padding,locations))
        if descs is None:
            sift_train.append([])
        else:
            sift_train.append(descs)
        i += 1
        if i%1000 == 0:
            print(i,datetime.now()-t)

    for img in X_test: 
        kps, descs = sift(img)
        #if len(img.shape) == 2 :
            #img = img[:,:,numpy.newaxis]
        hog_test.append(hog.compute(img,winStride,padding,locations))
        if descs is None:
            sift_test.append([])
        else:
            sift_test.append(descs)
        
    return hog_train, hog_test, sift_train, sift_test


def complete_sift(sift_train, sift_test):

    """
    Transform the descriptors of all key points into features using bag of words
    """
    
    #filt lines with null descriptors
    null_line = [i for i in range(len(sift_train)) if len(sift_train[i])==0]
    sift_train_filt = [sift_train[i] for i in range(len(sift_train)) if i not in null_line]
    
    #concatenate train and test
    sift_filt = numpy.concatenate((sift_train_filt,sift_test))
    
    #stack all the descriptors for train and test set
    descs = sift_filt[0]
    for idx, des in enumerate(sift_filt[1:]):
        descs = numpy.vstack((descs, des))

    #kmeans clustering
    k = 100
    voc, variances = kmeans(descs, k, 1)

    #calculate the histogram 
    n = len(sift_filt)
    n_train = len(sift_train_filt)
    sift_feature = np.zeros((n,k), "float32")
    for i in range(n):  
        words, distance = vq(sift_filt[i], voc)
        for w in words:
            sift_feature[i][w] += 1

    # Tf-Idf vectorizaiton
    nb_occurences = numpy.sum((sift_feature > 0) *1, axis=0)
    idf = numpy.array(numpy.log((1.0*n+1) / (1.0*nb_occurences + 1)), 'float32')

    # Scaling the words
    stdSlr = StandardScaler().fit(sift_feature)
    sift_feature = stdSlr.transform(sift_feature)

    return sift_feature[:n_train], sift_feature[n_train:]


if __name__ == '__main__':
    
    #label_unique, X_train, X_test, y_train, y_test = split()
    #hog_train, hog_test, sift_train, sift_test = feature_extractor(X_train, X_test)
    hog_train, hog_test, sift_train, sift_test = load_feature()
    sift_feature_train, sift_feature_test = complete_sift(sift_train, sift_test)
