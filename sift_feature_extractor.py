import cv2
from scipy.misc import imread

def sift(image):
    """
    reference:
    install opencv : https://www.scivision.co/compiling-opencv3-with-extra-contributed-modules/
    sift : http://www.pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/

    """
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    (kps, descs) = sift.detectAndCompute(image, None)
    #print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
    
    return kps, descs
    
if __name__ == '__main__':
    img = cv2.imread('data/image_0001.jpg')
    
    kps, descs = sift(img)


