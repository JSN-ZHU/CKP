import struct
import numpy as np
import os
import matplotlib.pyplot as plt

def load_mnist(path,kind='train'):
    """Load MNIST data from `path`"""

    labels_path = os.path.join(path,'%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

#使用了 https://www.jianshu.com/p/3d0bba113fd6 代码

x , y = load_mnist('C:/Users/DELL/Desktop/All_Files/Programs/MNIST')



'''
#fig, ax = plt.subplots(
#    nrows=2,
#    ncols=5,
#    sharex=True,
#    sharey=True, )
#
#ax = ax.flatten()
##for i in range(10):
#    img = x[y == i][0].reshape(28, 28)
#    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
#
#ax[0].set_xticks([])
#ax[0].set_yticks([])
#plt.tight_layout()
#plt.show()
#'''