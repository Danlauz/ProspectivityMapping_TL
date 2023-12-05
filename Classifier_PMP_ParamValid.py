# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:10:33 2023

@author: dany-
"""


import numpy as np


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV
from sys import stdout

def Classifier_PMP_Tuning(X_Train, Y_Train, seed, categ):
# List of element 
# 0 - Coord_X
# 1 - Coord_Y
# 2 - Residual magnetic field (high resolution)
# 3 - Vertical gradient of the residual magnetic field (high resolution)
# 4 - Magnetic analytical signal (high resolution)
# 5 - Bouguer anomaly (low resolution)
# 6 - Vertical gradient of gravity Anomaly (low resolution)
# 7 - Geological contact (boolean)
# 8 - Distance to geological contact
# 9 - Faults (boolean)
#10 - Distance to faults
#11 - Regional geology (gray image)
#12 - General geology (gray image)
#13 - Cupper mineralization

    
    
    scores = []
    params = []
    estimators = []
    
    if categ==1:
        nsets = 5
        models = RandomForestClassifier(random_state=seed)

        hparams = {'bootstrap':[False, True],
                     'criterion':['gini', 'entropy' ],
                     'max_depth': [2, 5, 10, 20, 30, 40, 50, None],
                     'max_features': ['auto', 'sqrt'],
                     'min_samples_leaf': [1, 2, 3, 4, 5],
                     'min_samples_split': [2, 3, 5, 10],
                     'n_estimators': [100, 200, 300, 400, 500, 600, 800, 1000],}
        
        grid_search = RandomizedSearchCV(models, hparams, n_iter=100, cv=5, n_jobs =-1, 
                                         scoring='roc_auc', error_score=False, return_train_score=True, random_state=seed)
        
        for n in range(nsets):
            # search best params
            search = grid_search.fit(X_Train[:,:,n], Y_Train[:,-1,n])
            params.append(search.best_params_)
            estimators.append(search.best_estimator_)
            scores.append(search.best_score_)
            
            percentage = (n+1)/nsets
            time_msg = "\rRunning Progress at {0:.2%} ".format(percentage)
            stdout.write(time_msg)
            
    else: 
        models = RandomForestRegressor(random_state=seed)

        hparams = {'bootstrap':[True],
                     'criterion':['squared_error'],
                     'max_depth': [ None],
                     'max_features': ['auto', 'sqrt'],
                     'min_samples_leaf': [1,],
                     'min_samples_split': [2],
                     'n_estimators': [100, 200, 300, 400, 500],}
        
        grid_search = RandomizedSearchCV(models, hparams, n_iter=10, cv=5, n_jobs =-1, 
                                         scoring='r2', error_score=False, return_train_score=True, random_state=seed)
        
        # search best params
        search = grid_search.fit(X_Train, Y_Train[:,-1])
        params.append(search.best_params_)
        estimators.append(search.best_estimator_)
        scores.append(search.best_score_)
        
    # choose the best
    best_id = np.argmax(scores)
    best_scores = np.max(scores)
    best_params = params[best_id]
    models = estimators[best_id]
    
    
    return  best_params, best_scores, models