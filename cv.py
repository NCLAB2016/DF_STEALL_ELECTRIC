#!/usr/bin/python
import numpy as np
import os
from six.moves import cPickle as pickle
from sklearn.linear_model import LogisticRegression 

# read in data
dataset = '/media/dat1/liao/TXAD/dataset'
train_data_filename = os.path.join(dataset, 'train.pickle')

with open(train_data_filename, 'rb') as f:
    train_data = pickle.load(f)
train_data = np.array(train_data, dtype=np.float32)
print ('dataset: ', train_data.shape)

# validation dataset size
n_folds = 10 
print ('select n_folds = ' + str(n_folds) + ', start to validation...')

import scipy as sp
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


train_pos = train_data[train_data[:,0]==1,:]
train_neg = train_data[train_data[:,0]==0,:]
np.random.shuffle(train_pos)
np.random.shuffle(train_neg)
num_pos = train_pos.shape[0] 
num_neg = train_neg.shape[0]
ratio_pos = num_pos / n_folds
ratio_neg = num_neg / n_folds

from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestClassifier
lls = list()
for i in range(n_folds):
    # split the dataset 
    valid_ = np.row_stack((train_pos[ratio_pos*i:ratio_pos*(i+1),:], 
                           train_neg[ratio_neg*i:ratio_neg*(i+1),:]))
    train_ = np.row_stack((train_pos[:ratio_pos*i,:], train_pos[ratio_pos*(i+1):,:],
                           train_neg[:ratio_neg*i,:], train_neg[ratio_neg*(i+1):,:]))

    ############### Here to define model ################
    #cls = LogisticRegression(n_jobs=-1)
    cls = RandomForestClassifier(n_estimators=2000,max_depth=10,verbose=1)
    model = cls.fit(train_[:,1:], train_[:,0])
    result = model.predict_proba(valid_[:,1:])
    ll = logloss(valid_[:,0], result[:,1])
    print ('fold ' + str(i) + ': logloss = ' + str(ll))
    lls.append(ll)

lls = np.array(lls)
print ('complete. mean logloss = {:.6f}, std = {:.6f}'.format(np.mean(lls), np.std(lls)))
