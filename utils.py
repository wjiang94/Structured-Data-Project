# import cv2
import Image
import numpy

# def load_data(folder_name='data/'):
#     img = cv2.imread(folder_name + 'image_0001.jpg')
#     print(img)
#     cv2.imshow('image',img)
    
def load_image(infilename) :
    img = Image.open(infilename)
    img.load()
#     img.show()
#     data = numpy.asarray(img, dtype="int32")
    data = numpy.asarray(img, dtype='float')
    return data

def save_image(npdata, outfilename) :
    img = Image.fromarray(numpy.asarray(numpy.clip(npdata, 0, 255), dtype='uint8'), 'RGB')
    img.save(outfilename)

if __name__ == '__main__':
    data = load_image('data/image_0002.jpg')
    print(data)
    print(data.shape)
#     save_image(data, 'data/image_0004.jpg')