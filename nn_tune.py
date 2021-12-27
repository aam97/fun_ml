'''
author: Ava Myers (aam97@pitt.edu)

Parameter tuning with Keras and XGBoost
'''

#===============================================
#Import libraries
#===============================================

#general python libraries
import pandas as pd
import importlib
import pickle
from time import time
from copy import deepcopy
import seaborn as sns
import numpy as np

seed = np.random.RandomState(6)

#ML libraries
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
cross_validation
#ANN class
from seqNN import *

#data preprocessor
from preprocess_data import *

#===============================================
#Get the dataset
#===============================================

sigName = 'ZHvvbb.npy'
bkgName = 'fullJZ.npy'

signal = load_npy(sigName, True)
bkg = load_npy(bkgName, False)

#concatenate and shuffle the data
data = np.concatenate([signal, bkg])
np.random.shuffle(data)

#get features
cols = ['gXENOISECUT_MET_et', 'gXERHO_MET_et', 'gXEJWOJ_MET_et', 'jXERHO_MET_et']

print('Training variables: {}'.format(cols))

#if we want to split dataset
x_train, x_test, y_train, y_test = split(data=data, test_size=0.2, cols=cols)

#if we want  to do cross validation
x = data[cols]
y = data['label']

x = x.astype([(name, '<f8') for name in x.dtype.names]).view(('<f8', len(x.dtype.names)))
#y = y.astype([(name, '<f8') for name in y.dtype.names]).view(('<f8', len(y.dtype.names)))

print(x.shape, y.shape)
print(x_train.shape, y_train.shape)

#CERN datasets are protected, so including open source dataset for sharing

#Using the Pima Indians diabetes database
#pima = pd.read_csv('diabetes.csv')
#print(pima.head())

#get the features, labels and convert to np arrays

#x, y = pima.values[:, 0:8], pima.values[:, 8]
#print(pima.shape, pima.shape)

#check percentage of both classes
#pima['Outcome'].value_counts()/pima.shape[0]

#===============================================
#XGBoost baseline
#===============================================

def xgb_baseline(x, y, params=None):
    t1 = time()
    clf = xgb.XGBClassifier(params)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, x, y, cv=cv)
    t2 = time()
    t= t2-t1
    
    print("Mean Accuracy: {:.2%}, Standard Deviation: {:.2%}".format(scores.mean(), scores.std()))
    print("Time taken: {:.2f} seconds".format(t))
    
    #test with some feature scaling
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)
    
    scores = cross_val_score(clf, x_std, y, cv=cv)
    
    print("Mean Accuracy: {:.2%}, Standard Deviation: {:.2%}".format(scores.mean(), scores.std()))

#run test with ANN

def ann(x, y, param_defaults, tuning_options=None):    
    acc = test(x=x, y=y, params=param_defaults)
    return acc
    
def get_xgb_defaults(learning_rate=0.1,
                     max_depth=10,
                     subsample=0.5,
                     colsample_bytree=0.5,
                     n_estimators=50,
                     objective='binary:logistic',
                     alpha=10,
                     gamma=10):

    defaults = {'learning_rate': learning_rate,
                'max_depth': max_depth,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'n_estimators': n_estimators,
                'objective': objective,
                'alpha': alpha,
                'gamma': gamma}

    return defaults

def test_xgb(x,y, params, tuning_options=None):

    if tuning_options:
        results={}

        for par, options, in tuning_options.items():
            results[par]={}
            params_tmp = deepcopy(params)

            for opt in options:
                print("\nEvaluating parameter \"{}\" using value \"{}\"...".format(par, opt))
                params_tmp[par]=opt

                try:
                    xgb_baseline(x, y, params_tmp)
                except Exception as e:
                    print('Error: {}, skipping'.format(e))
                    pass
    
#try XGBoost baseline
#xgb_baseline(x_train,y_train)

#param_defaults, params = get_defaults(), get_defaults()
param_defaults, params = get_xgb_defaults(), get_xgb_defaults()
print(param_defaults)

lr = [0.0001, 0.001, 0.01, 0.1, 1.]
max_depth = [1, 10, 20, 50]
subsample = [0.1, 0.2, 0.5, 0.7, 1.]
colsample_bytree = [0.1, 0.2, 0.5, 0.7, 1.]
n_estimators = [10, 50, 100, 500, 1000]
objective = ["binary:logistic"]
alpha = [0.1, 1, 10, 20]
gamma = [0.1, 1, 10, 20]

tuning_opts = get_tuning_opt(learning_rate=lr,
                             max_depth=max_depth,
                             subsample=subsample,
                             colsample_bytree=colsample_bytree,
                             n_estimators=n_estimators,
                             objective=objective,
                             alpha=alpha,
                             gamma=gamma)

test_xgb(x=x, y=y, params=param_defaults, tuning_options=tuning_opts)
'''n_units = [4, 8, 16, 32, 64]
#n_units = [4]
n_layers = [2, 4, 8, 16, 32]
#n_layers = [2]
loss = ['binary_crossentropy', 'categorical_crossentropy', 'sparse_categorical_crossentropy']
#loss = ['binary_crossentropy']
initializer = ['random_uniform', 'random_normal', 'TruncatedNormal', 'glorot_normal', 'glorot_uniform']
#initializer = ['random_uniform']
learning_rate = [0.0001, 0.001, 0.01, 0.1, 1]
#learning_rate = [0.0001]
optimizer = ['Adam', 'Adamax', 'Adagrad', 'Sgd', 'RMSprop']
#optimizer = ['Adam']
epochs = [5, 10, 20, 40, 80]
#epochs = [100]
batch_size = [1, 2, 4, 8, 16]
#batch_size = [16]
one_hot = [True]

tuning_options = get_tuning_opt(n_units=n_units,
                                n_layers=n_layers,
                                loss=loss, 
                                initializer=initializer,
                                learning_rate=learning_rate, 
                                optimizer=optimizer, 
                                epochs=epochs, 
                                batch_size=batch_size,
                                one_hot=one_hot)

print(tuning_options)

print("================================\nSequential model baseline\n==========================")
baseline = test(x=x, y=y, params=param_defaults)

print("================================\nTuning parameters\n==========================")
results = test(x=x_train, y=y_train, params=param_defaults, tuning_options=tuning_options)'''

#save results in a pickle file so they can be plotted later
with open('cross_validation_results.pkl', 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
    print("\n Results saved successfully!")
'''

with open('cross_validation_results.pkl', 'rb') as f:
    results = pickle.load(f)

df = transform_results(results)
visualize_dist(df, save_figure=True)
visualize_trend(df, save_figure=True)
'''
