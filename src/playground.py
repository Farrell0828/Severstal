import pandas as pd 
from sklearn.model_selection import StratifiedKFold 
import cv2 as cv 
from matplotlib import pyplot as plt 
from PIL import Image 
import numpy as np 

df = pd.read_csv('./data/train_new.csv', index_col=0)
rle = df.loc['008ef3d74.jpg'].EncodedPixels_1
print(rle)

height= 256
width = 1600

mask= np.zeros(width*height, dtype=np.uint8)
array = np.asarray([int(x) for x in rle.split()])
starts = array[0::2]-1
lengths = array[1::2]
for index, start in enumerate(starts):
    mask[int(start):int(start+lengths[index])] = 1
mask = mask.reshape(width, height).T
print(mask.shape)
img = Image.fromarray(mask*255, mode='L')
print(np.array(img).shape)
img = img.resize((516, 128), resample=0)
img.show()
arr = np.array(img)
print(arr.dtype)
print(arr.shape)
one_count = (arr == 0).sum()
zero_count = (arr == 255).sum()
print(one_count, zero_count)
print(one_count + zero_count == 516 * 128)