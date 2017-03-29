import cv2

def sift(img_name):
    """
    reference:
    install opencv : https://www.scivision.co/compiling-opencv3-with-extra-contributed-modules/
    sift : http://www.pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/

    """

    image = cv2.imread(img_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    (kps, descs) = sift.detectAndCompute(gray, None)
    print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
    return kps, descs
    
if __name__ == '__main__':
    img = 'data/image_0001.jpg'
    
    kps, descs = sift(img)
