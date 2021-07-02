import json
import cv2
import numpy as np
import matplotlib.pyplot as plt


def parse_json(d):
    """ Data annotation processing
    Args:
        d: dict, annotation information dict of a sample photo
    Returns:
        arr: array, annotation information
    """
    arr = np.array([
        d['top'], d['height'], d['left'],  d['width'], d['label']
    ])
    arr = arr.astype(int)
    return arr

# load all train data infomation 
train_json = json.load(open('../input/mchar_train.json'))

img = cv2.imread('../input/mchar_train/000000.png')
arr = parse_json(train_json['000000.png'])

plt.figure(figsize=(10, 10))
plt.subplot(1, arr.shape[1]+1, 1)
plt.imshow(img)
plt.xticks([])
plt.yticks([])

for idx in range(arr.shape[1]):
    plt.subplot(1, arr.shape[1]+1, idx+2)
    plt.imshow(img[arr[0, idx]:arr[0, idx]+arr[1, idx],
               arr[2, idx]:arr[2, idx]+arr[3, idx]])
    plt.title(arr[4, idx])
    plt.xticks([])
    plt.yticks([])

plt.show()