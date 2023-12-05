# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:06:40 2023

@author: dany-
"""

import pandas as pd
import numpy as np
from numpy import load

from Functional import Dimension, InputDataZone

from numpy import save
import matplotlib.pyplot as plt


from Figure_PMP import Figure_PMP
from Classifier_PMP import Classifier_PMP as C_PMP
from sklearn.metrics import brier_score_loss, log_loss, matthews_corrcoef

Feuillet = "F33FG"

X_Train = load("ProcessData\\TrainingData\\" + Feuillet + "_Data.npy") 
X_Test = load("ProcessData\\ValidationData\\" + Feuillet + "_Data.npy")
  
Y_Train = load("ProcessData\\TrainingData\\" + Feuillet + "_Loc.npy") 
Y_Test = load("ProcessData\\ValidationData\\" + Feuillet + "_Loc.npy") 

X_TrainRigth = load("ProcessData\\TrainingData\\" + Feuillet + "_DataOther.npy") 
Y_TrainRigth = load("ProcessData\\ValidationData\\" + Feuillet + "_DataOther.npy") 
  
X_TrainAll= load("ProcessData\\TrainingData\\" + Feuillet + "_DataAll.npy") 
Y_TrainAll = load("ProcessData\\ValidationData\\" + Feuillet + "_DataAll.npy")  

pd_ID= pd.read_csv("ProcessData\\InputData" + "\\" + Feuillet + ".csv")
pd_OD= pd.read_csv("ProcessData\\OccurenceData\\" + Feuillet + "_"+ "Cu_P" +  ".csv")
pd_ODAll= pd.read_csv("ProcessData\\OccurenceData\\" + Feuillet + "_"+ "Cu_PS" +  ".csv")

from Functional import AdaptatedSpatialAssociation as ASA
from Functional import Dist2Mine as D2M
from Functional import ComputeDistanceToMineralization as D2Mine

from Classifier_PMP_ParamValid import Classifier_PMP_Tuning
from ECDF import ecdf

seed=562

        
GeoASA = ASA(pd_OD, pd_ID, Feuillet)

DistMine=D2Mine(pd_ID, pd_OD)
dM=D2M(pd_OD, Feuillet)
DistGeo=(GeoASA<=np.quantile(GeoASA,0.25))

plt.figure().set_figwidth(10)
fig, ax = plt.subplots(figsize=(6,7),facecolor='w')
qe, pe = ecdf(GeoASA)
plt.plot(qe,pe)
ax.set_xlim(0, 5)

Ind=(DistGeo)*(DistMine>dM) #(GeoASA==0)*
Ind[Ind>0]=1

fea=[0,1,2,3,4,
     5,6,7,  8,9,10,
     11,
     13,14,15,16,   24,25,26, 27,28,29,30,31,32,33,34,35,36,]
     #37,38,39,40,41,42,43,44,45,46,47,48] 

AllData=pd_ID.iloc[:, [x+2 for x in fea] ]

Xmin, Xmax, Ymin, Ymax, Lat, Long = Dimension(Feuillet)  

Index= ((Lat-pd_OD['Coord_Y']-1)*Long+pd_OD['Coord_X']).to_numpy()
Index=(np.rint(Index)).astype(int)

IndexAll= ((Lat-pd_ODAll['Coord_Y']-1)*Long+pd_ODAll['Coord_X']).to_numpy()
IndexAll=(np.rint(IndexAll)).astype(int)

nbsim = np.size(X_Train,axis=2)

LongData=pd_ID['Coord_X'].to_numpy().reshape(Lat,Long)/(Long-1)*(Xmax-Xmin)+Xmin
LatData=pd_ID['Coord_Y'].to_numpy().reshape(Lat,Long)/(Lat-1)*(Ymax-Ymin)+Ymin


Data=GeoASA.reshape(Lat,Long)
Data[Data<=np.quantile(GeoASA,0.25)]=0
plt.figure().set_figwidth(10)
fig, ax = plt.subplots(figsize=(6,7), facecolor='w')
im = ax.pcolormesh( LongData, LatData, Data, shading='auto', cmap='turbo', vmin=0., vmax=9)
plt.plot(pd_OD[pd_OD['Mine']==1]['Coord_X']/(Long-1)*(Xmax-Xmin)+Xmin,pd_OD[pd_OD['Mine']==1]['Coord_Y']/(Lat-1)*(Ymax-Ymin)+Ymin,"*k", markersize=14) 
plt.plot(pd_OD[pd_OD['Deposit']==1]['Coord_X']/(Long-1)*(Xmax-Xmin)+Xmin,pd_OD[pd_OD['Deposit']==1]['Coord_Y']/(Lat-1)*(Ymax-Ymin)+Ymin,"ok", markersize=13) 
plt.plot(pd_OD[pd_OD['Occurence']==1]['Coord_X']/(Long-1)*(Xmax-Xmin)+Xmin,pd_OD[pd_OD['Occurence']==1]['Coord_Y']/(Lat-1)*(Ymax-Ymin)+Ymin,"^k", markersize=12) 
plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.044, pad=0.1) 
ax.set_xlabel('Longitude', fontsize=22)
ax.set_ylabel('Latitude', fontsize=22)
ax.set_xlim(-79.5, -78.5)
ax.set_ylim(48, 48.5)
ax.set_box_aspect(1)
plt.show() 

Data=DistMine.reshape(Lat,Long)
plt.figure().set_figwidth(10)
fig, ax = plt.subplots(figsize=(6,7), facecolor='w')
im = ax.pcolormesh( LongData, LatData, Data, shading='auto', cmap='turbo', vmin=0., vmax=50)
plt.plot(pd_OD[pd_OD['Mine']==1]['Coord_X']/(Long-1)*(Xmax-Xmin)+Xmin,pd_OD[pd_OD['Mine']==1]['Coord_Y']/(Lat-1)*(Ymax-Ymin)+Ymin,"*k", markersize=14) 
plt.plot(pd_OD[pd_OD['Deposit']==1]['Coord_X']/(Long-1)*(Xmax-Xmin)+Xmin,pd_OD[pd_OD['Deposit']==1]['Coord_Y']/(Lat-1)*(Ymax-Ymin)+Ymin,"ok", markersize=13) 
plt.plot(pd_OD[pd_OD['Occurence']==1]['Coord_X']/(Long-1)*(Xmax-Xmin)+Xmin,pd_OD[pd_OD['Occurence']==1]['Coord_Y']/(Lat-1)*(Ymax-Ymin)+Ymin,"^k", markersize=12) 
plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.044, pad=0.1) 
ax.set_xlabel('Longitude', fontsize=22)
ax.set_ylabel('Latitude', fontsize=22)
ax.set_xlim(-79.5, -78.5)
ax.set_ylim(48, 48.5)
ax.set_box_aspect(1)
plt.show() 

Data=Ind.reshape(Lat,Long)
plt.figure().set_figwidth(10)
fig, ax = plt.subplots(figsize=(6,7), facecolor='w')
im = ax.pcolormesh( LongData, LatData, Data, shading='auto', cmap='turbo', vmin=0., vmax=1)
plt.plot(pd_OD[pd_OD['Mine']==1]['Coord_X']/(Long-1)*(Xmax-Xmin)+Xmin,pd_OD[pd_OD['Mine']==1]['Coord_Y']/(Lat-1)*(Ymax-Ymin)+Ymin,"*k", markersize=14) 
plt.plot(pd_OD[pd_OD['Deposit']==1]['Coord_X']/(Long-1)*(Xmax-Xmin)+Xmin,pd_OD[pd_OD['Deposit']==1]['Coord_Y']/(Lat-1)*(Ymax-Ymin)+Ymin,"ok", markersize=13) 
plt.plot(pd_OD[pd_OD['Occurence']==1]['Coord_X']/(Long-1)*(Xmax-Xmin)+Xmin,pd_OD[pd_OD['Occurence']==1]['Coord_Y']/(Lat-1)*(Ymax-Ymin)+Ymin,"^k", markersize=12) 
plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.044, pad=0.1) 
plt.plot(Y_Test[Y_Test[:,-1,0]==0,0,0]/(Long-1)*(Xmax-Xmin)+Xmin,Y_Test[Y_Test[:,-1,0]==0,1,0]/(Lat-1)*(Ymax-Ymin)+Ymin,"sw", markersize=4)  
plt.plot(Y_Train[Y_Train[:,-1,0]==0,0,0]/(Long-1)*(Xmax-Xmin)+Xmin,Y_Train[Y_Train[:,-1,0]==0,1,0]/(Lat-1)*(Ymax-Ymin)+Ymin,"sw", markersize=4)
ax.set_xlabel('Longitude', fontsize=22)
ax.set_ylabel('Latitude', fontsize=22)
ax.set_xlim(-79.5, -78.5)
ax.set_ylim(48, 48.5)
ax.set_box_aspect(1)
plt.show() 

Data=Ind.reshape(Lat,Long)
plt.figure().set_figwidth(10)
fig, ax = plt.subplots(figsize=(6,7), facecolor='w')
im = ax.pcolormesh( LongData, LatData, Data, shading='auto', cmap='turbo', vmin=0., vmax=1)
plt.plot(pd_OD[pd_OD['Mine']==1]['Coord_X']/(Long-1)*(Xmax-Xmin)+Xmin,pd_OD[pd_OD['Mine']==1]['Coord_Y']/(Lat-1)*(Ymax-Ymin)+Ymin,"*k", markersize=14) 
plt.plot(pd_OD[pd_OD['Deposit']==1]['Coord_X']/(Long-1)*(Xmax-Xmin)+Xmin,pd_OD[pd_OD['Deposit']==1]['Coord_Y']/(Lat-1)*(Ymax-Ymin)+Ymin,"ok", markersize=13) 
plt.plot(pd_OD[pd_OD['Occurence']==1]['Coord_X']/(Long-1)*(Xmax-Xmin)+Xmin,pd_OD[pd_OD['Occurence']==1]['Coord_Y']/(Lat-1)*(Ymax-Ymin)+Ymin,"^k", markersize=12) 
plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.044, pad=0.1) 
plt.plot(Y_Test[Y_Test[:,-1,2]==0,0,8]/(Long-1)*(Xmax-Xmin)+Xmin,Y_Test[Y_Test[:,-1,8]==0,1,2]/(Lat-1)*(Ymax-Ymin)+Ymin,"sw", markersize=4)  
plt.plot(Y_Train[Y_Train[:,-1,2]==0,0,8]/(Long-1)*(Xmax-Xmin)+Xmin,Y_Train[Y_Train[:,-1,8]==0,1,2]/(Lat-1)*(Ymax-Ymin)+Ymin,"sw", markersize=4)
ax.set_xlabel('Longitude', fontsize=22)
ax.set_ylabel('Latitude', fontsize=22)
ax.set_xlim(-79.5, -78.5)
ax.set_ylim(48, 48.5)
ax.set_box_aspect(1)
plt.show() 

Y_Test[:,0,:]=Y_Test[:,0,:]/(Long-1)*(Xmax-Xmin)+Xmin
Y_Test[:,1,:]=Y_Test[:,1,:]/(Lat-1)*(Ymax-Ymin)+Ymin
Y_Train[:,0,:]=Y_Train[:,0,:]/(Long-1)*(Xmax-Xmin)+Xmin
Y_Train[:,1,:]=Y_Train[:,1,:]/(Lat-1)*(Ymax-Ymin)+Ymin

_, _, _, _, LongMin, LongMax, LatMin, LatMax = InputDataZone(Feuillet, pd_ID, Long, Lat, Xmin, Xmax, Ymin, Ymax)

  
# %% Training Algorithm and predict on local zone
clf_all=[]
acc_all=[]
Proba_all=[]
Class_all=[]
bs_all=[]
log_loss_all=[]
mcc_all=[]

best_params, best_scores, models = Classifier_PMP_Tuning(X_Train, Y_Train, seed, categ=1)  
print(best_params) 

for T in range(0,nbsim):
    clf, accuracy, ProbaTest, ClassTest = C_PMP(X_Train[:,fea,T],X_Test[:,fea,T],Y_Train[:,:,T],Y_Test[:,:,T], best_params , seed)  
    bs=brier_score_loss(Y_Test[:,-1,T], ProbaTest[:,1])
    logloss=log_loss(Y_Test[:,-1,T]==1, ProbaTest[:,1])
    mcc=matthews_corrcoef(Y_Test[:,-1,T], ClassTest)
    
    clf_all.append(clf) 
    acc_all.append(accuracy) 
    Proba_all.append(ProbaTest) 
    Class_all.append(ClassTest) 
    bs_all.append(bs)
    log_loss_all.append(logloss)
    mcc_all.append(mcc)
    
print( (np.mean(acc_all),np.std(acc_all)) )    
print( (np.mean(bs_all),np.std(bs_all)) )
print( (np.mean(log_loss_all),np.std(log_loss_all)) )
print( (np.mean(mcc_all),np.std(mcc_all)) )



predicted=[]
for T in range(0, nbsim):         
    p = clf_all[T].predict_proba(AllData.to_numpy())
    p=np.reshape(p[:,1],(-1,1))
    predicted.append(p)

pMean=np.array(predicted).mean(axis=0)
pStd=np.array(predicted).std(axis=0)

save("Results\\prediction\\" + Feuillet + "_Data.npy", predicted)
save("Results\\acc\\" + Feuillet + "_Data.npy", acc_all)
save("Results\\bs\\" + Feuillet + "_Data.npy", bs_all)
save("Results\\class\\" + Feuillet + "_Data.npy", Class_all)
save("Results\\clf\\" + Feuillet + "_Data.npy", clf_all)
save("Results\\proba\\" + Feuillet + "_Data.npy", Proba_all)
save("Results\\best_param\\" + Feuillet + "_Data.npy", best_params)
             
Figure_PMP(Feuillet, pd_ID, pd_OD, pd_ODAll, X_Test, Y_Test, Y_Train, clf_all, Proba_all, Class_all, fea, pMean, pStd, predicted )
    
# %% Training Algorithm  on all other zone and predict on local zone

pd_ID= pd.read_csv("ProcessData\\InputData" + "\\" + Feuillet + ".csv")
pd_OD= pd.read_csv("ProcessData\\OccurenceData\\" + Feuillet + "_"+ "Cu_P" +  ".csv")
pd_ODAll= pd.read_csv("ProcessData\\OccurenceData\\" + Feuillet + "_"+ "Cu_PS" +  ".csv")


clf_all=[]
acc_all=[]
Proba_all=[]
Class_all=[]
bs_all=[]    
log_loss_all=[]
mcc_all=[]

best_params, best_scores, models = Classifier_PMP_Tuning(X_TrainAll, Y_TrainAll, seed, categ=1)      
print(best_params)                      
for T in range(0,nbsim):
    
    clf, accuracy, ProbaTest, ClassTest = C_PMP(X_TrainAll[:,fea,T],X_Test[:,fea,T],Y_TrainAll[:,:,T],Y_Test[:,:,T], best_params , seed)  
    bs=brier_score_loss(Y_Test[:,-1,T], ProbaTest[:,1])
    logloss=log_loss(Y_Test[:,-1,T], ProbaTest[:,1])
    mcc=matthews_corrcoef(Y_Test[:,-1,T], ClassTest)
    
    clf_all.append(clf) 
    acc_all.append(accuracy) 
    Proba_all.append(ProbaTest) 
    Class_all.append(ClassTest) 
    bs_all.append(bs)
    log_loss_all.append(logloss)
    mcc_all.append(mcc)
    
print( (np.mean(acc_all),np.std(acc_all)) )    
print( (np.mean(bs_all),np.std(bs_all)) )
print( (np.mean(log_loss_all),np.std(log_loss_all)) )
print( (np.mean(mcc_all),np.std(mcc_all)) )

predicted=[]
for T in range(0, nbsim):         
    p = clf_all[T].predict_proba(AllData.to_numpy())
    p=np.reshape(p[:,1],(-1,1))
    predicted.append(p)
                
pMean=np.array(predicted).mean(axis=0)
pStd=np.array(predicted).std(axis=0)

save("Results\\prediction\\" + Feuillet + "_DataAll.npy", predicted)
save("Results\\acc\\" + Feuillet + "_DataAll.npy", acc_all)
save("Results\\bs\\" + Feuillet + "_DataAll.npy", bs_all)
save("Results\\class\\" + Feuillet + "_DataAll.npy", Class_all)
save("Results\\clf\\" + Feuillet + "_DataAll.npy", clf_all)
save("Results\\proba\\" + Feuillet + "_DataAll.npy", Proba_all)
save("Results\\best_param\\" + Feuillet + "_DataAll.npy", best_params)
   
Figure_PMP(Feuillet, pd_ID, pd_OD, pd_ODAll, X_Test, Y_Test, Y_Train, clf_all, Proba_all, Class_all, fea, pMean, pStd, predicted )

# %%


for x in range(2,45):
    Data=pd_ID.iloc[:,x].to_numpy().reshape(Lat,Long)
    plt.figure().set_figwidth(10)
    fig, ax = plt.subplots(figsize=(6,7),facecolor='w')
    im = ax.pcolormesh( LongData, LatData, Data, cmap='turbo', shading='auto', vmin=np.quantile(Data,0.01), vmax=np.quantile(Data,0.99))
    plt.plot(pd_ODAll[pd_ODAll['Mine']==1]['Coord_X'],pd_ODAll[pd_ODAll['Mine']==1]['Coord_Y'],"*w", markersize=14, mec = 'k') 
    plt.plot(pd_ODAll[pd_ODAll['Deposit']==1]['Coord_X'],pd_ODAll[pd_ODAll['Deposit']==1]['Coord_Y'],"ow", markersize=13, mec = 'k') 
    plt.plot(pd_ODAll[pd_ODAll['Occurence']==1]['Coord_X'],pd_ODAll[pd_ODAll['Occurence']==1]['Coord_Y'],"^w", markersize=12, mec = 'k') 
    plt.plot(pd_OD[pd_OD['Mine']==1]['Coord_X'],pd_OD[pd_OD['Mine']==1]['Coord_Y'],"*k", markersize=14) 
    plt.plot(pd_OD[pd_OD['Deposit']==1]['Coord_X'],pd_OD[pd_OD['Deposit']==1]['Coord_Y'],"ok", markersize=13) 
    plt.plot(pd_OD[pd_OD['Occurence']==1]['Coord_X'],pd_OD[pd_OD['Occurence']==1]['Coord_Y'],"^k", markersize=12) 
    #plt.plot(Y_Test[Y_Test[:,-1,0]==0,0,0],Y_Test[Y_Test[:,-1,0]==0,1,0],"sk", markersize=12)  
    #plt.plot(Y_Train[Y_Train[:,-1,0]==0,0,0],Y_Train[Y_Train[:,-1,0]==0,1,0],"sk", markersize=12)
    plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.044) 
    ax.set_xlabel('Longitude', fontsize=22)
    ax.set_ylabel('Latitude', fontsize=22)
    if x==17:
        ax.set_title('R_AUI', fontsize=22)
    else:
        ax.set_title(pd_ID.columns[x], fontsize=22)
    ax.set_box_aspect(1)
    plt.show() 

# %%     
Ind[Ind>0]=1
Data=Ind.reshape(Lat,Long)
plt.figure().set_figwidth(10)
fig, ax = plt.subplots(figsize=(6,7),facecolor='w')
im = ax.pcolormesh( LongData, LatData, Data, shading='auto', cmap='turbo', vmin=0., vmax=1)
plt.plot(Y_Test[Y_Test[:,-1,0]==0,0,0],Y_Test[Y_Test[:,-1,0]==0,1,0],"sw", markersize=6)  
plt.plot(Y_Train[Y_Train[:,-1,0]==0,0,0],Y_Train[Y_Train[:,-1,0]==0,1,0],"sw", markersize=6)
plt.plot(pd_OD[pd_OD['Mine']==1]['Coord_X'],pd_OD[pd_OD['Mine']==1]['Coord_Y'],"*k", markersize=12) 
plt.plot(pd_OD[pd_OD['Deposit']==1]['Coord_X'],pd_OD[pd_OD['Deposit']==1]['Coord_Y'],"ok", markersize=11) 
plt.plot(pd_OD[pd_OD['Occurence']==1]['Coord_X'],pd_OD[pd_OD['Occurence']==1]['Coord_Y'],"^k", markersize=10) 
plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.044) 
ax.set_xlabel('Longitude', fontsize=22)
ax.set_ylabel('Latitude', fontsize=22)
ax.set_box_aspect(1)
plt.show()     




