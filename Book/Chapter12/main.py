import os
import struct 
import numpy as np 
import matplotlib.pyplot as plt

os.chdir('/home/cameron/Dropbox/University/PhD/Teaching/COMP219-AI/COMP219/Book/Chapter12/')

def load_mnist(path, kind='train'):
    """Load MNIST data from path"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte' % kind)
    print(os.getcwd())
    print(labels_path)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',  # big endian two unsigned ints
                                 lbpath.read(8))
        labels = np.fromfile(lbpath, dtype='uint8')

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(
                                                    len(labels), 784)
        images = ((images / 255.) - .5) * 2

    return images, labels

X_train, y_train = load_mnist('', kind='t10k')
print('Rows: {0}, Columns: {1}'.format(X_train.shape[0], X_train.shape[1]))

# fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
# ax = ax.flatten()
# for i in range(10):
#     img = X_train[y_train==i][0].reshape(28,28)
#     ax[i].imshow(img, cmap='Greys')
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()

np.savez('mnist_scaled')