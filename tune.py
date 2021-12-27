import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

from preprocess_data import *

rcParams['figure.figsize'] = 12, 4

def xg_reg():
    file = "~/Desktop/Dataset/Training/Features_Variant_1.csv"
    
    df = pd.read_csv(file, header=None)
    df.sample(n=5)
    
    print("Dataset has {} entries and {} features".format(*df.shape))
    
    X, y = df.loc[:,:52].values, df.loc[:,53].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    #"learn" the mean from training data
    mean_train = np.mean(y_train)
    
    #get predictions from test set
    baseline_pred = np.ones(y_test.shape)*mean_train
    
    #compute Mean Absolute Error (MAE)
    mae_baseline = mean_absolute_error(y_test, baseline_pred)
    
    print("Baseline MAE = {:.2f}".format(mae_baseline))
    
    #setup parameters dictrionary
    
    params = {'max_depth':6,
              'min_child_weight': 1,
              'eta': 0.3,
              'subsample': 1,
              'colsample_bytree': 1,
              'objective': 'reg:linear',
    }
    
    params['eval_metric'] = "mae"
    
    num_boost_round = 999
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtest, "Test")],
        early_stopping_rounds=10
    )
    
    print("Best AUC = {:.2f} with {} rounds". format(
        model.best_score,
        model.best_iteration+1))
    
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )
    
    print(cv_results)
    print(cv_results['test-mae-mean'].min())
    
    #test combinations of max_depth, min_child_weight parameters
    
    '''grid_search_params = [
    (max_depth, min_child_weight)
    for max_depth in range(9,12)
    for min_child_weight in range(5,8)
    ]
    
    #define initial best params and MAE
    
    min_mae = float("Inf")
    best_params = None
    
    for max_depth, min_child_weight in grid_search_params:
    print("CV with max_depth={}, min_child_weight={}".format(
    max_depth,
    min_child_weight))
    
    #update parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    
    #run CV
    cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=42,
    nfold=5,
    metrics={'mae'},
    early_stopping_rounds=10
    )
    
    #update best MAE
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    
    if mean_mae < min_mae:
    min_mae = mean_mae
    best_params = (max_depth, min_child_weight)
    
    print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))'''
    
    #params['max_depth']=best_params[0]
    params['max_depth']=10
    #params['min_child_weight']=best_params[1]
    params['min_child_weight']=6
    
    best_params = None
    
    '''grid_search_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(7,11)]
    for colsample in [i/10. for i in range(7,11)]
    ]
    
    min_mae= float("Inf")
    
    #start with largest values and go down to smallest
    for subsample, colsample in reversed(grid_search_params):
    print("CV with subsample={}, colsample={}".format(subsample, colsample))
    
    #update parameters
    params['subsample'] = subsample
    params['colsample_by_tree'] = colsample
    
    #run cv
    cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=42,
    nfold=5,
    metrics={'mae'},
    early_stopping_rounds=10
    )
    
    #update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    
    if mean_mae < min_mae:
    min_mae = mean_mae
    best_params = (subsample, colsample)
    
    print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))'''
    
    params['subsample'] = 0.9
    params['colsample_bytree'] = 1.0
    
    
    '''min_mae = float("Inf")
    
    for eta in [0.3, 0.2, 0.1, 0.05, 0.01, 0.005]:
    print("CV with eta={}".format(eta))
    
    params['eta'] = eta
    
    cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=42,
    nfold=5,
    metrics={'mae'},
    early_stopping_rounds=10
    )
    
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
    min_mae = mean_mae
    best_params = eta
    
    print("Best params: {}, MAE: {}".format(best_params, min_mae))'''
    
    params['eta'] = 0.01
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtest, "Test")],
        early_stopping_rounds=10
    )

    print("Best AUC: {:.2f} in {} rounds".format(model.best_score, model.best_iteration+1))

def optimize(dtrain):

    num_boost_round = 999
    
    params = {'max_depth':6,
              'min_child_weight': 1,
              'eta': 0.3,
              'subsample': 1,
              'colsample_bytree': 1,
              'objective': 'binary:logistic',
    }    
    params['eval_metric'] = "auc"
    grid_search_params = [
        (max_depth, min_child_weight)
        for max_depth in range(9,12)
        for min_child_weight in range(5,8)
    ]
    
    #define initial best params and AUC
    
    max_auc = 0.0
    best_params = None
    
    for max_depth, min_child_weight in grid_search_params:
        print("CV with max_depth={}, min_child_weight={}".format(
            max_depth,
            min_child_weight))
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics={'auc'},
            early_stopping_rounds=10
        )
        
        #update best AUC
        mean_auc = cv_results['test-auc-mean'].min()
        boost_rounds = cv_results['test-auc-mean'].argmin()
        
        print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))
        
        if mean_auc > max_auc:
            max_auc = mean_auc
            best_params = (max_depth, min_child_weight)
    
            print("Best params: {}, {}, AUC: {}".format(best_params[0], best_params[1], max_auc))
            #update parameters
            params['max_depth'] = best_params[0]
            params['min_child_weight'] = best_params[1]

    best_params = None
    grid_search_params = [
        (subsample, colsample)
        for subsample in [i/10. for i in range(7,11)]
                for colsample in [i/10. for i in range(7,11)]
    ]
    
    max_auc= 0.0
    
    #start with largest values and go down to smallest
    for subsample, colsample in reversed(grid_search_params):
        print("CV with subsample={}, colsample={}".format(subsample, colsample))
        if mean_auc > max_auc:
            max_auc = mean_auc
            best_params = (subsample, colsample)
            print("Best params: {}, {}, AUC: {}".format(best_params[0], best_params[1], max_auc))
            
            #update parameters
            params['subsample'] = best_params[0]
            params['colsample_by_tree'] = best_params[1]
            
            #run cv
            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                seed=42,
                nfold=5,
                metrics={'auc'},
                early_stopping_rounds=10
            )
            
    max_auc = 0.0
    
    for eta in [0.3, 0.2, 0.1, 0.05, 0.01, 0.005]:
        print("CV with eta={}".format(eta))
    
        params['eta'] = eta
        
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics={'auc'},
            early_stopping_rounds=10
        )
        
        mean_auc = cv_results['test-auc-mean'].min()
        boost_rounds = cv_results['test-auc-mean'].argmin()
        print("\tAUC {} for {} rounds\n".format(mean_auc, boost_rounds))
        if mean_auc > max_auc:
            max_auc = mean_auc
            best_params = eta    

    params['eta'] = best_params

    #print best parameters
    print(params)
            
def xgb_clf():

    sigName = 'ZHvvbb.npy'
    bkgName = 'fullJZ.npy'

    signal = load_npy(sigName, True)
    bkg = load_npy(bkgName, False)

    data = np.concatenate([signal, bkg])
    np.random.shuffle(data)
    
    cols = ['gXENOISECUT_MET_et', 'gXERHO_MET_et', 'gXEJWOJ_MET_et', 'jXERHO_MET_et', 'gXEJWOJ_MET_mst', 'gXEJWOJ_MET_mht', 'EventVariables_RhoA', 'EventVariables_RhoB', 'EventVariables_RhoC' ]

    x_train, x_test, y_train, y_test = split(data=data, test_size=0.1, cols=cols)
    
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    optimize(dtrain)
    
    '''model = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27,
        use_label_encoder=False
    )

    xgb_param = model.get_xgb_params()
    print(xgb_param)
    
    cv_result = xgb.cv(xgb_param, dtrain, num_boost_round=model.get_params()['n_estimators'], nfold=5, metrics='auc', early_stopping_rounds=50)
    model.set_params(n_estimators=cv_result.shape[0])

    print("Fitting model...")
    model.fit(x_train, y_train, eval_metric='auc')

    train_pred = model.predict(x_train)
    train_pred_prob = model.predict_proba(x_train)[:,1]

    print("\nModel Report")
    print("Accuracy: %.4g" % metrics.accuracy_score(y_train, train_pred))
    print("AUC score (train): %f" % metrics.roc_auc_score(y_train, train_pred_prob))

    #feat_imp = pd.Series(model.booster.get_fscore()).sort_values(ascending=False)
    print(model.feature_importances_)
    #plot
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    #feat_imp.plot(kind='bar', title='Feature importance')
    plt.ylabel('Feature importance score')

    plt.show()'''
    
xgb_clf()
