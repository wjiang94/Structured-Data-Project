import os
import numpy
from scipy.misc import imread
from sklearn.model_selection import train_test_split

"""
split all the images into train and test set and transfer all the labels into one-hot representation
"""

def one_hot(n_classes, y):
    return numpy.eye(n_classes)[y]

def split(test_size=0.2, path='101_ObjectCategories/'):
    
    img = []
    label_unique = []
    label = []
    i = 0

    """retreive all the images and labels"""
    
    for filename in os.listdir(path):
        if filename != '.DS_Store':
            label_unique.append(filename)
            for imgname in os.listdir(path+filename):
                if imgname != '.DS_Store':
                    img.append(imread(path+filename+'/'+imgname))
                    label.append(i) 
            i += 1

    y = one_hot(len(label_unique),label)
    
    X_train, X_test, y_train, y_test = train_test_split(img, y, test_size=test_size)
    
    return label_unique, X_train, X_test, y_train, y_test
        
if __name__ == '__main__':
    
    label_unique, X_train, X_test, y_train, y_test = split()
