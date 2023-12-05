# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:10:33 2023

@author: dany-
"""
from sklearn.ensemble import RandomForestClassifier

def Classifier_PMP(X_Train, X_Val, Y_Train, Y_Val, best_params, seed):

# Split the data into a training set and a test set
    train_data=X_Train
    train_labels=Y_Train[:,-1]
    
    test_data=X_Val
    test_labels=Y_Val[:,-1]

# Create the supervised classifier
    clf= RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                min_samples_split=best_params['min_samples_split'],
                                min_samples_leaf=best_params['min_samples_leaf'],
                                max_features=best_params['max_features'], max_depth=best_params['max_depth'],
                                criterion=best_params['criterion'], bootstrap=best_params['bootstrap'],
                                      n_jobs=-1, random_state=seed)


# Fit the classifier to the labeled data
    clf.fit(train_data, train_labels)

# Predict the labels of the tresting data
    p = clf.predict_proba(test_data)
    ClassPred= clf.predict(test_data)
    accuracy = clf.score(test_data, test_labels)
    #FeaturesImportance=Pimp(clf, test_data, test_labels)

    return clf, accuracy, p, ClassPred