# *_*coding:utf-8 *_*
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import config1


def plot(config=config1, index=0):
    name = config['name']
    path = config['path']
    files = [os.path.join(path, f) for f in os.listdir(path)]
    img = cv2.imread(files[index], cv2.COLOR_BGR2GRAY)
    plt.imshow(img, cmap='gray')
    plt.title('image originale')
    plt.show()
    return img


def get_sum(img, axis=0):
    res = np.sum(img, axis=axis)
    plt.plot(range(res.size), res)
    plt.title('sum selon l\'axis {}'.format(axis))
    plt.xlabel('pixel')
    plt.ylabel('valeur')
    plt.show()


if __name__ == "__main__":
    get_sum(plot())
    get_sum(plot(), 1)
