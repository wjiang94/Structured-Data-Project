from train_test_split import split
from hog_feature_extractor import HOGFeatureExtractor
from sift_feature_extractor import sift
from datetime import datetime

def feature_extractor(X_train, X_test):
    
    """
    return hog and sift features for both train and test set
    """
    
    hog_train = []
    hog_test = []
    sift_train = []
    sift_test = []
    hog = HOGFeatureExtractor()
    
    i = 0
    for img in X_train:
        kps, descs = sift(img)
        if len(img.shape) == 2 :
            img = img[:,:,numpy.newaxis]
        hog_train.append(hog.calc_gradient_for_image(img, False))
        if descs is None:
            sift_train.append([])
        else:
            sift_train.append(descs)
        i += 1
        if i%100 == 0:
            print(i)

    for img in X_test: 
        kps, descs = sift(img)
        if len(img.shape) == 2 :
            img = img[:,:,numpy.newaxis]
        hog_test.append(hog.calc_gradient_for_image(img, False))
        if descs is None:
            sift_train.append([])
        else:
            sift_train.append(descs)
        i += 1
        if i%100 == 0:
            print(i)
        
    return hog_train, hog_test, sift_train, sift_test

if __name__ == '__main__':
    t = datetime.now()
    label_unique, X_train, X_test, y_train, y_test = split()
    hog_train, hog_test, sift_train, sift_test = feature_extractor(X_train, X_test)
    print(datetime.now()-t)
