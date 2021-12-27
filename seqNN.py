'''
author: Ava Myers (aam97@pitt.edu)

Class definition for sequential neural network model and related functions

'''

#===============================================
#Import libraries
#===============================================

import pandas as pd
import importlib
import pickle
from time import time
from copy import deepcopy
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

seed = np.random.RandomState(6)

#ML libraries
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical

#===============================================
#Base class for sequential neural network model
#===============================================

class SequentialNN:

    #---------------------------------------------------------------
    #NN constructor
    #---------------------------------------------------------------
    
    def __init__(self, input_dim, n_layers, n_units,
                 activation, activation_out, loss, initializer, optimizer, learning_rate, epochs, batch_size, metrics=['accuracy'], one_hot=False):
        """
        Params:
        input_dim: (int) number of features
        n_layers: (int) number of layers of the model (excluding the input layer)
        n_units: (list) number of units in each layer(excluding the input layer)
        activation: (str) activation function used in all layers except output
        activation_out: (str) activation function used in output layer
        loss: (str) loss functon
        initializer: (str) kernel initializer
        optimizer: (str) optimizer
        metrics: (list of strings) metrics used
        epochs: (int) number of epochs to train for
        batch_size: (int) number of samples per batch
        one_hot: (bool) whether one hot encoding is needed
        """

        self.input_dim = input_dim
        self.n_layers = n_layers
        self.n_units = [n_units, 1]
        self.activation = activation
        self.activation_out = activation_out
        self.loss = loss
        self.initializer = initializer
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size
        self.one_hot = one_hot

        #initialize the model as sequential
        self.model = Sequential()

    #---------------------------------------------------------------
    #Function to build up the model: adds layers and compile
    #---------------------------------------------------------------

    def build_model(self):
        
        #ensure n_units list's length is the same as n_layers
        if self.n_layers != len(self.n_units):
            #expand list by repeating number of nodes (minus output layer)
            n_nodes, n_nodes_out = self.num_units[0], self.num_units[-1]
            self.n_units = [i for i in range(self.n_layers-1) for i in [n_nodes]]
            self.n_units.append(n_nodes_out)
            
        #loop through layers
        for i in range(self.n_layers):
            
            #different setups for each layer
            if i == 0: #input and first hidden layer
                self.model.add(Dense(units=self.n_units[i],
                                     input_dim=self.input_dim,
                                     kernel_initializer=self.initializer,
                                     activation=self.activation))
                
            elif i+1 == self.n_layers: #output layer
                self.model.add(Dense(units=self.n_units[i],
                                     kernel_initializer=self.initializer,
                                     activation=self.activation_out))
                
            else: #hidden layers
                self.model.add(Dense(units=self.n_units[i],
                                     kernel_initializer=self.initializer,
                                     activation=self.activation))
                
        #instantiate the optimizer
        optimizer_class = getattr(importlib.import_module("tensorflow.keras.optimizers"),
                                  self.optimizer)
        self.optimizer = optimizer_class(lr=self.learning_rate)
        
        #compile model
        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=self.metrics)

    #---------------------------------------------------------------
    #Function to evaluate model - using cross-validation
    #---------------------------------------------------------------

    def eval(self, x, y, n_splits=10):
        
        '''
        Inputs:
        --------------------
        x: (np.array) features
        y: (np.array) labels
        n_splits: (int) number of folds for the cross-validation
        
        Returns:
        --------------------
        mean_accuracy: (float) the average accuracy based on the cross-validation.
        '''
        
        score_list = []
        t1 = time()

        print("Starting {}-fold cross-validation...".format(n_splits))
        
        kfold = StratifiedKFold(n_splits=n_splits,
                                shuffle=True,
                                random_state=6)
        
        #loop through different folds
        for train_index, test_index in kfold.split(x,y):
            
            #one-hot encoding
            if self.one_hot:
                y_one_hot = to_categorical(y)
            else:
                y_one_hot = y
                
            #fit the model
            self.model.fit(x[train_index],
                           y_one_hot[train_index],
                           epochs=self.epochs,
                           batch_size=self.batch_size)
            
            scores = self.model.evaluate(x[test_index],
                                         y_one_hot[test_index])
            
            #accuracy is second array element
            score_list.append(scores[1])
            
        t2 = time()
        t = (t2 - t1)/60. #convert time to minutes

        print("Finished cross-valiation in {:.1f} mintues.".format(t))

        #convert score list to np array and calculate mean, std
        score_list = np.array(score_list)
        mean_acc = score_list.mean()
        std_acc = score_list.std()

        print("Mean Accuracy: {:.2%}, Standard Deviation: {:.2%}".format(mean_acc, std_acc))
        return mean_acc

#---------------------------------------------------------------
#Function to return default list of hyperparameters
#---------------------------------------------------------------

def get_defaults(input_dim=4,
                 n_layers=2,
                 n_units=8,
                 activation='relu',
                 activation_out='sigmoid',
                 loss='binary_crossentropy',
                 initializer='random_uniform',
                 optimizer='Adam',
                 learning_rate=0.001,
                 metrics=['accuracy'],
                 epochs=10,
                 batch_size=4,
                 one_hot=True):
    
    """
    Returns a dict of default hyperparameter values  
    """
    
    defaults = {'input_dim': input_dim, 
                'n_layers': n_layers, 
                'n_units': n_units, 
                'activation': activation, 
                'activation_out': activation_out, 
                'loss': loss, 
                'initializer': initializer, 
                'optimizer': optimizer, 
                'learning_rate': learning_rate, 
                'metrics': metrics, 
                'epochs': epochs, 
                'batch_size': batch_size, 
                'one_hot': one_hot}
    
    return defaults

#---------------------------------------------------------------
#Function to define tuning options
#---------------------------------------------------------------

def get_tuning_opt(**kwargs):
    '''
    Inputs:
    ------------
    Keyword arguments - the key words can be any of the following:
    input_dim, n_layers, n_units, activation, activation_out, 
    loss, initializer, optimizer, learning_rate, epochs, batch_size, 
    metrics=['accuracy'], one_hot=False
    
    Returns:
    -------------
    tuning_options: Python dict made up of the given keyword, argument pairs
    '''
    
    tuning_options = {}
    
    for param, options in kwargs.items():
        tuning_options[param] = options 
        
    return tuning_options

#---------------------------------------------------------------
#Builds a NN model with given parameters and returns cross-validation accuracy
#---------------------------------------------------------------

def build_eval(x, y, params):
    '''
    Inputs:
    -----------
    x: (np.array) the features
    y: (np.array) the labels
    param_dict: Python dict containing the tuning parameters and values
    
    Returns:
    ------------
    result: (float) percentage accuracy based on cross-validation
    '''
    
    model = SequentialNN(input_dim=params['input_dim'], 
                         n_layers=params['n_layers'], 
                         n_units=params['n_units'],
                         activation=params['activation'], 
                         activation_out=params['activation_out'], 
                         loss=params['loss'], 
                         initializer=params['initializer'], 
                         optimizer=params['optimizer'], 
                         learning_rate=params['learning_rate'],
                         epochs=params['epochs'],
                         batch_size=params['batch_size'])
    model.build_model()
    result = model.eval(x, y)
        
    return result
#---------------------------------------------------------------
#Test function
#---------------------------------------------------------------
def test(x, y, params, tuning_options=None):
    '''
    Inputs:
    -----------
    X: (np.array) the features
    y: (np.array) the labels
    param_dict: Python dict containing the tuning parameters and values
    tuning_options: Python dict made up of the given keyword, argument pairs
    
    Returns:
    -----------
    results: a dict when tuning_options is provided, otherwise a float that's the mean accuracy.
    '''         
    if tuning_options:
        results = {}
        
        for par, options in tuning_options.items():
            results[par] = {}
            params_tmp = deepcopy(params)
            
            for opt in options:
                print("\nEvaluating parameter \"{}\" using value \"{}\"...".format(par, opt))
                #update parameter
                params_tmp[par] = opt
                
                try:
                    results[par][opt] = build_eval(x, y, params_tmp)
                    
                except Exception as e:
                    results[par][opt] = 'NaN'
                    print('Error: {}, skipping'.format(e))
                    pass
                
        return results
    else:
        return build_eval(x, y, params)

#---------------------------------------------------------------
#Transforms results dictionary into df (better for analysis)
#---------------------------------------------------------------
def transform_results(results):
    '''
    Inputs:
    ------------
    results: (dict) results returned by build_eval
    
    Returns:
    ------------
    df_plot: (pandas df) wrangled long format dataframe
    
    '''
    
    df = pd.DataFrame(results)
    
    #get col names
    val_vars = df.columns.tolist()
    
    #reset index, rename index col
    df = df.reset_index().rename(columns={'index': 'option'})
    
    #transform from wide to long format (for plotting)
    df_long = pd.melt(df, id_vars='option', value_vars=val_vars)
    df_long = df_long.rename(columns={'variable': 'parameter'})
    
    #exclude null values
    df_long = df_long[~df_long['value'].isnull()]
    df_long= df_long.query("value!=0 & value!='NaN'")
    
    #calculate range and std of each parameter group, convert to df
    ranges = df_long.groupby('parameter').apply(lambda grp: grp.value.max() - grp.value.min())
    std = df_long.groupby('parameter').apply(lambda grp: grp.value.std())
    spread = pd.concat([ranges, std], axis=1).rename(columns={0: 'ranges', 1:'std'})
    
    df_spread = pd.merge(df_long, spread, how='left', left_on='parameter', right_index=True)
    
    #reorder columns
    df_spread = df_spread[['parameter', 'option', 'value', 'ranges', 'std']]
    
    #remove null values
    df_spread = df_spread.query('ranges!=0')
    
    #change dtype
    df_spread['value'] - df_spread['value'].astype(float)
    
    #sort df and use resulting index to slice (ensures plot is ordered accordingly)
    idx = df_spread.sort_values(by=['ranges', 'option'], ascending=False).index
    df_plot = df_spread.loc[idx, :]
    
    return df_plot

#---------------------------------------------------------------
#Returns the best set of options given a dataframe
#---------------------------------------------------------------

def get_best_options(df):
    
    '''
    Inputs:
    -------------
    df: the results df returned by transform_results function
    
    Returns:
    -------------
    best_opt: a list of list containing param and value
    
    '''
    
    best_opt = df.groupby('parameter').apply(lambda grp: grp.nlargest(1, 'value'))[['parameter', 'option']].values.tolist()
    
    return best_opt

#---------------------------------------------------------------
#Visualizes resulting distributions with boxplot and swarmplot
#---------------------------------------------------------------
def visualize_dist(df, save_figure=False):
    
    
    '''
    Inputs:
    -------------
    df: (pandas df) results df returned by transform_results
    save_figure: (bool) whether to save figure
    
    '''
    
    fig, axis = plt.subplots(figsize=(16,12))
    df['value'] - df['value'].astype(float)
    
    sns.boxplot(x='parameter', y='value', data=df,  Ax=axis)
    sns.swarmplot(x='parameter', y='value', data=df, size=12, ax=axis)
    axis.set_xlabel('Parameters', size=16)
    axis.set_ylabel('Values', size=16)
    fig_title = 'Parameter tuning results distribution'
    axis.set_title(fig_title, y=1.05, fontsize=30)
    
    if save_figure:
        fig_name = fig_title + '.pdf'
        fig.savefig(fig_name)
        
#---------------------------------------------------------------
#Visualizes resulting trend with lineplot
#---------------------------------------------------------------
def visualize_trend(df, save_figure=False):
    
    '''
    Inputs:
    -------------
    df: (pandas df) results df returned by transform_results
    save_figure: (bool) whether to save figure
    
    '''
    
    fig, axes = plt.subplots(nrows=2, ncols=4, sharey=True, figsize=(30,12))
    axes = axes.flatten()
    
    #get parameter list
    params = df.parameter.unique().tolist()
    
    #loop through axes
    for i, axis in enumerate(axes):
        try:
            #one subplot per parameter
            param = params[i]
            df_param = df.query("parameter==@param")
            df_param.plot(kind='line', x='option', y='value', ax=axis)
            
            #put learning rate on logx scale to avoid confusion
            if param=='learning_rate':
                axis.set_xscale('log')
                
            #set x ticks, tick labels
            if param in ('initializer', 'optimizer'):
                axis.set_xticks(np.arange(5))
                axis.set_xticklabels(df_param.option)
                
            axis.set_xlabel(param, fontsize=16)
            axis.set_ylabel('Accuracy', fontsize=16)
            axis.get_legend().remove()
            
        except:
            #remove last axis since we only have 7 parameters
            fig.delaxes(axis)
            
    fig_title = 'Parameter Tuning Trend'
    fig.suptitle(fig_title, y=0.92, verticalalignment='bottom', fontsize=30)
    
    plt.show()
    
    if(save_figure):
        fig_name = fig_title + '.pdf'
        fig.savefig(fig_name)
        
#======================================================
