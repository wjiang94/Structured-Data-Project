import numpy

from utils import load_image

# References:
# http://mccormickml.com/2013/05/09/hog-person-detector-tutorial/
# http://www.learnopencv.com/histogram-of-oriented-gradients/
class HOGFeatureExtractor:
    """
    nbins: number of bins that will be used
    unsigned: if True the sign of the angle is not considered
    """
    def __init__(self, nbins=9, unsigned=True):
        self.nbins = nbins
        self.unsigned = unsigned

    def _calc_gradient_for_channel(self, I, unflatten):
        nX, nY = I.shape
        histogram = numpy.zeros((4, 4, self.nbins))

        for i in range(0, nX):
            for j in range(0, nY):
                dx, dy = 0, 0
                if i < nX - 1:
                    dx += I[i + 1, j]
                if i > 0:
                    dx -= I[i - 1, j]
                if j < nY - 1:
                    dy += I[i, j + 1]
                if j > 0:
                    dy -= I[i, j - 1]

                if dy == 0 and dx == 0:
                    continue

                magnitude = numpy.sqrt(dx**2 + dy**2)
                if self.unsigned:
                    if dx == 0:
                        angle = numpy.pi / 2
                    else:
                        angle = numpy.arctan(dy / dx)
                    angle = (angle + numpy.pi / 2) / (numpy.pi / self.nbins)
                else:
                    angle = numpy.arctan2(dx, dy)
                    angle = (angle + numpy.pi) / (2 * numpy.pi / self.nbins)

                bin_pos = int(numpy.floor(angle))
                # handle corner case
                if bin_pos == self.nbins:
                    bin_pos = 0
                    angle = 0

                closest_bin = bin_pos

                if bin_pos == 0:
                    if angle < 0.5:
                        second_closest_bin = self.nbins - 1
                    else:
                        second_closest_bin = 1
                elif bin_pos == self.nbins - 1:
                    if angle < self.nbins - 0.5:
                        second_closest_bin = self.nbins - 2
                    else:
                        second_closest_bin = 0
                else:
                    if angle < bin_pos + 0.5:
                        second_closest_bin = bin_pos - 1
                    else:
                        second_closest_bin = bin_pos + 1

                # closest_bin_distance + second_closest_bin_distance = 1
                if angle < bin_pos + 0.5:
                    second_closest_bin_distance = angle - (bin_pos - 0.5)
                else:
                    second_closest_bin_distance = (bin_pos + 1.5) - angle

                r = second_closest_bin_distance
                histogram[i * 4 / nX, j * 4 / nY, closest_bin] += r * magnitude
                histogram[i * 4 / nX, j * 4 / nY, second_closest_bin] += (1 - r) * magnitude

        ret = numpy.zeros((3, 3, self.nbins * 4))

        for i in range(3):
            for j in range(3):
                aux = histogram[i:i + 2, j:j + 2, :].flatten().copy()
                aux = aux / numpy.linalg.norm(aux)
                ret[i, j, :] = aux

        if unflatten:
            ret.reshape(9, -1)
        return ret.flatten()

    def calc_gradient_for_image(self, I, unflatten):
        nchannels = I.shape[2]
        ret = []

        for i in range(nchannels):
            ret.append(self._calc_gradient_for_channel(I[:,:,i], unflatten))

        if unflatten:
            return numpy.array(ret).reshape(nchannels * 9, -1)
        return numpy.array(ret).flatten()

if __name__ == '__main__':
    img = load_image('data/image_0001.jpg')
    
    hog = HOGFeatureExtractor()
    print(hog.calc_gradient_for_image(img, False))
