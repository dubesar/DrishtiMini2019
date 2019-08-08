#!/usr/bin/env python
# coding: utf-8

# In[86]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder

def get_prepare(source=None,stand = False,norm =False,test_size = 0.3, random_state = 15,add_ones=False):
    
    df = pd.read_csv('{}'.format(source))
    
    features = df.columns[1:-1]
    target = df.columns[-1]
    print('\nFeatures: \n',features.values)
    print('\nTarget: \n',target,'\n\n')
    
    df.drop(columns=['ID'],inplace=True)
    
    ## removing non g/h values from class
    print('\n\nRemoving incorrect values from class')
    a = df.columns.get_loc('{}'.format(target))
    positions=[]
    for i in range(1,df.shape[0]):
        if not (df.iloc[i,a] == 'g'or df.iloc[i,a] == 'h'):
            positions.append(i)
    print('done')
    
    ## removing non float values from features
    print('\n\nRemoving non float values from features....')
    for i in features:
        a = df.columns.get_loc('{}'.format(i))
        for j in range(1,df['{}'.format(i)].shape[0]):
            if ('{}'.format(df.iloc[j,a]).isalnum() or '{}'.format(df.iloc[j,a]).isalpha()) and (not df.iloc[j,a] ==0) :
                if j not in positions:
                    positions.append(j)

    ## trying to convert strings to float. If an error occurs the position is noted 
    objects = []
    for i in df.columns:
        if df[i].dtype == 'object' and not i == target:
            if j not in positions:
                objects.append(i)

    for i in objects:
        for j in range(df[i].shape[0]):
            try:
                float(df['{}'.format(i)].iloc[j])
            except:
                if j not in positions:
                    positions.append(j)
        df['{}'.format(i)] = pd.to_numeric(df['{}'.format(i)],errors='coerce')

    print('done\n\nA total of {} rows were removed. Their positions are :\n\n'.format(np.shape(np.unique(positions))[0]),positions)
    df.drop(positions,inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True,inplace=True)
    del positions
    
    for c in features: 
        if stand: df['{}'.format(c)]= (df['{}'.format(c)] - df['{}'.format(c)].mean())/ df['{}'.format(c)].std()                             # Standardization 
        if norm : df['{}'.format(c)]= (df['{}'.format(c)] - min(df['{}'.format(c)]))/ (max(df['{}'.format(c)])-min(df['{}'.format(c)]))      # Normalizing
    
    if add_ones:
        ones = np.ones((df.shape[0],1))
        df.insert(0,'ones',ones)
    
    df = np.array(df)
    
    x = df[:,0:(np.shape(df)[1]-1)]
    x=x.astype('float64')
    
    y = df[:,-1]
    y =LabelEncoder().fit_transform(y)
    y=y.astype('float64')
    
    x = np.array(x)
    y = np.array(y)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state,shuffle=True)
    y_train,y_test = y_train.reshape((y_train.shape[0],1)),y_test.reshape((y_test.shape[0],1))
    print('\n\nx_train.shape, x_test.shape, y_train.shape, y_test.shape =',x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return df,features,target,x_train, x_test, y_train, y_test


# $\large data= \frac{data-mean(data)}{std(data)}$  # Standardization
# 
# $\large data= \frac{data-min(data)}{max(data)-min(data)}$   # Normalizing

# In[93]:


# df,features,target,x_train, x_test, y_train,` y_test = get_prepare('Magic_Telescope_data.csv',stand = True,norm =True,test_size = 0.3, random_state = 15,add_ones=True)

