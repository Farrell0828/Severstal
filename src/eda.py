#%%
import pandas as pd 
import numpy as np 
from PIL import Image 
import cv2 
from matplotlib import pyplot as plt 

#%%
def rle2maskResize(rle, d_height, d_width):
    # CONVERT RLE TO MASK 
    if (pd.isnull(rle))|(rle==''): 
        return np.zeros((d_height, d_width), dtype=np.uint8)
    height= 256
    width = 1600
    mask= np.zeros(width*height, dtype=np.uint8)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]-1
    lengths = array[1::2]
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
    mask = mask.reshape(width, height).T
    img = Image.fromarray(mask, mode='L').resize((d_width, d_height), resample=0)
    return np.array(img)

#%%
train_df = pd.read_csv('./data/train_new.csv', index_col=0)
train_df.head()

#%%
rles_1 = train_df[~pd.isna(train_df['EncodedPixels_1'])]['EncodedPixels_1']
rles_2 = train_df[~pd.isna(train_df['EncodedPixels_2'])]['EncodedPixels_2']
rles_3 = train_df[~pd.isna(train_df['EncodedPixels_3'])]['EncodedPixels_3']
rles_4 = train_df[~pd.isna(train_df['EncodedPixels_4'])]['EncodedPixels_4']
#%%
def get_sizes(mask):
    sizes = []
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    for c in range(1, num_component):
        p = (component == c)
        sizes.append(p.sum())
    return sizes
#%%
mask = rle2maskResize(rles_1[0], 256, 1600)
plt.imshow(mask)

#%%
num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
for c in range(1, num_component):
    p = (component == c)
    print(p.sum())
    plt.figure()
    plt.imshow(p)

#%%
sizes_1 = []
for rle in rles_1:
    sizes_1 += get_sizes(rle2maskResize(rle, 256, 1600))

#%%
len(sizes_1)

#%%
import seaborn as sns

#%%
plt.plot(sizes_1)

#%%
sizes_2 = []
for rle in rles_2:
    sizes_2 += get_sizes(rle2maskResize(rle, 256, 1600))

sizes_3 = []
for rle in rles_3:
    sizes_3 += get_sizes(rle2maskResize(rle, 256, 1600))

sizes_4 = []
for rle in rles_4:
    sizes_4 += get_sizes(rle2maskResize(rle, 256, 1600))

#%%
plt.hist(sizes_1)

#%%
sizes_1 = np.array(sizes_1)
sizes_2 = np.array(sizes_2)
sizes_3 = np.array(sizes_3)
sizes_4 = np.array(sizes_4)

#%%
print(np.median(sizes_1))
print(np.median(sizes_2))
print(np.median(sizes_3))
print(np.median(sizes_4))

#%%
print(np.mean(sizes_1))
print(np.mean(sizes_2))
print(np.mean(sizes_3))
print(np.mean(sizes_4))

#%%
print(np.min(sizes_1))
print(np.min(sizes_2))
print(np.min(sizes_3))
print(np.min(sizes_4))

#%%
