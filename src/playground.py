#%%
import pandas as pd 
from sklearn.model_selection import StratifiedKFold 
import cv2 as cv 
from matplotlib import pyplot as plt 
from PIL import Image 
import numpy as np 

#%%
df = pd.read_csv('./data/train_new.csv', index_col=0)
df.head(10)

#%%
df['EncodedPixels_1'].str.split()
#%%
def calc_n(x):
    if isinstance(x, float): return np.nan
    sum = 0
    for n in x[1::2]:
        sum += int(n)
    return sum 
#%%
df['EncodedPixels_4'].str.split().apply(calc_n).mean()

#%%
df['n1'] = df['EncodedPixels_1'].str.split().apply(calc_n)
df['n2'] = df['EncodedPixels_2'].str.split().apply(calc_n)
df['n3'] = df['EncodedPixels_3'].str.split().apply(calc_n)
df['n4'] = df['EncodedPixels_4'].str.split().apply(calc_n)

#%%
n_df = df[['n1', 'n2', 'n3', 'n4']]

#%%
n_df.info()

#%%
n_df.summary()

#%%
n_df.describe()

#%%
import seaborn as sns
from matplotlib import pyplot as plt 

#%%
sns.distplot(n_df[n_df['n1'].notnull()]['n1'])

#%%
sns.distplot(n_df[n_df['n2'].notnull()]['n2'])

#%%
sns.distplot(n_df[n_df['n3'].notnull()]['n3'])

#%%
sns.distplot(n_df[n_df['n4'].notnull()]['n4'])

#%%
plt.figure()
sns.distplot(n_df[n_df['n1'].notnull()]['n1'], hist_kws={'log':True})
sns.distplot(n_df[n_df['n2'].notnull()]['n2'], hist_kws={'log':True})
sns.distplot(n_df[n_df['n3'].notnull()]['n3'], hist_kws={'log':True})
sns.distplot(n_df[n_df['n4'].notnull()]['n4'], hist_kws={'log':True})

#%%
n_df['n1'].quantile([0.05, 0.1, 0.15, 0.2])

#%%
n_df['n2'].quantile([0.05, 0.1, 0.15, 0.2])

#%%
n_df['n3'].quantile([0.05, 0.1, 0.15, 0.2])

#%%
n_df['n4'].quantile([0.05, 0.1, 0.15, 0.2])

#%%
import keras.backend as K

#%%
a = K.variable(np.random.randn(2, 8, 8, 2))
b = K.variable(np.random.randn(2, 8, 8, 2))
a

#%%
K.sum(a, [1, 2], keepdims=True)


#%%
K.equal(K.sum(a, [1, 2], keepdims=True), 0)

#%%
mask = K.variable([[True, False], [True, False]])
K.eval(mask)

#%%
a*K.cast(mask, 'float32')

#%%
import numpy as np 
a = np.random.randn(2, 8, 8, 2)
b = np.array([[True, False], [False, False]]).reshape(2, 1, 1, 2)

#%%
a*b
#%%
c = a*b
c[1, :, :, 0]

#%%
import numpy as np 
a = np.random.randn(2, 8, 8, 3)
a.argmax()

#%%
