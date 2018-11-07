import numpy as np
from PIL import Image
import pandas as pd
from itertools import combinations



def train_df_aug(train_df, aub_num=5000):
    train_df_new = pd.DataFrame(columns=train_df.columns.tolist())
    train_df_new['images'] = train_df_new['images'].astype(object)
    train_df_new['masks'] = train_df_new['masks'].astype(object)
    for i in range(5):
        train_df_small = train_df[(train_df.z > i*200) & (train_df.z < (1+i)*200)]
        count = 0
        for id1, id2 in combinations(train_df_small.index.tolist(),2):
            print(count)
            count += 1
            img1 = np.asarray(train_df_small.loc[id1, 'images'])
            img2 = np.asarray(train_df_small.loc[id2, 'images'])
            mask1 = np.asarray(train_df_small.loc[id1, 'masks'])
            mask2 = np.asarray(train_df_small.loc[id2, 'masks'])
            id_new = id1+'_'+id2
            img_new = np.zeros(img1.shape)
            mask_new = np.zeros(mask1.shape)
            z_new = 0
            coverage_new = 0
            coverage_class_new = 0
            mask_union = np.logical_or(mask1,mask2)
            mask_new[mask_union] = 255
            alpha = np.random.rand()
            img_new[np.tile(~mask_union[:,:,np.newaxis],(1,1,3))] = alpha*img1[np.tile(~mask_union[:,:,np.newaxis],(1,1,3))] + (1-alpha)*img2[np.tile(~mask_union[:,:,np.newaxis],(1,1,3))]
            img_new[np.tile(np.logical_and(mask_union,mask1)[:,:,np.newaxis],(1,1,3))] = img1[np.tile(np.logical_and(mask_union,mask1)[:,:,np.newaxis],(1,1,3))]
            img_new[np.tile(np.logical_and(mask_union,mask2)[:,:,np.newaxis],(1,1,3))] = img2[np.tile(np.logical_and(mask_union,mask2)[:,:,np.newaxis],(1,1,3))]
            if np.sum(np.logical_and(mask_union,mask1))>np.sum(np.logical_and(mask_union,mask2)):  
                z_new = train_df_small.loc[id1,'z']
                coverage_new = train_df_small.loc[id1,'coverage']
                coverage_class_new = train_df_small.loc[id1,'coverage_class']
            else:
                z_new = train_df_small.loc[id2,'z']
                coverage_new = train_df_small.loc[id2,'coverage']
                coverage_class_new = train_df_small.loc[id2,'coverage_class']
            train_df_new.loc[id_new, 'z'] = z_new
            train_df_new.loc[id_new, 'coverage'] = coverage_new
            train_df_new.loc[id_new, 'coverage_class'] = coverage_class_new
            train_df_new.loc[id_new, 'images'] = Image.fromarray(img_new.astype(np.uint8))
            train_df_new.loc[id_new, 'masks'] = Image.fromarray(mask_new.astype(np.uint8))
            if count>1000:
                break
    train_df = train_df.append(train_df_new)
    return train_df