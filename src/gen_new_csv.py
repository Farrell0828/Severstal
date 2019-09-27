
import pandas as pd 
import numpy as np 

df = pd.read_csv('./data/train.csv')
df.head(20)

df['ImageId'] = df['ImageId_ClassId'].map(lambda x: x.split('.')[0]+'.jpg')
new_df = pd.DataFrame({'ImageId':df['ImageId'][::4]})
new_df['EncodedPixels_1'] = df['EncodedPixels'][::4].values
new_df['EncodedPixels_2'] = df['EncodedPixels'][1::4].values
new_df['EncodedPixels_3'] = df['EncodedPixels'][2::4].values
new_df['EncodedPixels_4'] = df['EncodedPixels'][3::4].values
for i in range(4):
    new_df['dt_{}'.format(i+1)] = (~new_df['EncodedPixels_{}'.format(i+1)].isna()).astype(int)
new_df['dt_0'] = (new_df[['dt_1', 'dt_2', 'dt_3', 'dt_4']].sum(axis=1) == 0).astype(int)
new_df.reset_index(inplace=True, drop=True)
new_df.fillna('', inplace=True); 
new_df['MaskCount'] = np.sum(new_df.iloc[:,1:5] != '', axis=1).values

new_df.to_csv('./data/train_new_cls.csv', index=False)

