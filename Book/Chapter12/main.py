import os
import struct 
import numpy as np 

os.chdir('/home/cameron/Dropbox/University/PhD/Teaching/COMP219-AI/Book/Chapter12')

def load_mnist(path, kind='train'):
    """Load MNIST data from path"""
    labels_path = os.path.join(path,
                               '%s-label-idx1-ubyte' % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte' % kind)
    print(labels_path)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',  # big endian two unsigned ints
                                 lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(
                                                    len(labels), 784)
        images = ((images / 255.) - .5) * 2

    return images, labels

X_train, y_train = load_mnist('', kind='train')