# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:35:07 2023

@author: dany-
"""


import numpy as np
from Functional import Dimension, InputDataZone, OutputDataZone

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn import metrics

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from scipy import interpolate
    
def Figure_PMP(Feuillet, pd_ID, pd_OD, pd_ODAll, X_Test, Y_Test, Y_Train, clf_all, Proba_all, Class_all, fea, pMean, pStd, predicted ):

    

    Xmin, Xmax, Ymin, Ymax, Lat, Long = Dimension(Feuillet) 
    
    _, idxV_P = OutputDataZone(Feuillet, pd_OD, Long, Lat)
    
    IsMine=pd_OD.copy()
    IsMine=IsMine.to_numpy()
    IsMine=IsMine[idxV_P,:]
    IsMine=IsMine[IsMine[:,2]==1,:]
    Mine=IsMine[:,3]
    Gite=IsMine[:,4]
    Indice=IsMine[:,5]
                
    IsMineIdx= (Lat-IsMine[: ,1]-1)*Long + IsMine[: ,0]
    IsMineIdx=(np.rint(IsMineIdx)).astype(int)
    IsMineIdx=np.reshape(IsMineIdx,(-1,1))
    
    pM=pMean[IsMineIdx]
    pS=pStd[IsMineIdx]
    print(np.round([np.mean(pM[Mine==1]),np.mean(pS[Mine==1]),np.mean(pM[Gite==1]),
                np.mean(pS[Gite==1]),np.mean(pM[Indice==1]),np.mean(pS[Indice==1]),np.mean(pMean[pMean>0.5]),np.mean(pStd[pMean>0.5])], 2))
    
    _, idxV_PS = OutputDataZone(Feuillet, pd_ODAll, Long, Lat)
    
    IsMineAll=pd_ODAll.copy()
    IsMineAll=IsMineAll.to_numpy()
    IsMineAll=IsMineAll[idxV_PS,:]
    IsMineAll=IsMineAll[IsMineAll[:,2]==1,:]
    Mine=IsMineAll[:,3]
    Gite=IsMineAll[:,4]
    Indice=IsMineAll[:,5]
                
    IsMineAllIdx= (Lat-IsMineAll[: ,1]-1)*Long + IsMineAll[: ,0]
    IsMineAllIdx=(np.rint(IsMineAllIdx)).astype(int)
    IsMineAllIdx=np.reshape(IsMineAllIdx,(-1,1))
    
    pM=pMean[IsMineAllIdx]
    pS=pStd[IsMineAllIdx]
    print(np.round([np.mean(pM[Mine==1]),np.mean(pS[Mine==1]),np.mean(pM[Gite==1]),
                np.mean(pS[Gite==1]),np.mean(pM[Indice==1]),np.mean(pS[Indice==1])], 2))
        
    SMALL_SIZE = 18
    matplotlib.rc('font', size=SMALL_SIZE)
    matplotlib.rc('axes', titlesize=SMALL_SIZE)

    LongData=pd_ID['Coord_X'].to_numpy().reshape(Lat,Long)/(Long-1)*(Xmax-Xmin)+Xmin
    LatData=pd_ID['Coord_Y'].to_numpy().reshape(Lat,Long)/(Lat-1)*(Ymax-Ymin)+Ymin

    pd_OD['Coord_X']=pd_OD['Coord_X'].to_numpy()/(Long-1)*(Xmax-Xmin)+Xmin
    pd_OD['Coord_Y']=pd_OD['Coord_Y'].to_numpy()/(Lat-1)*(Ymax-Ymin)+Ymin

    pd_ODAll['Coord_X']=pd_ODAll['Coord_X'].to_numpy()/(Long-1)*(Xmax-Xmin)+Xmin
    pd_ODAll['Coord_Y']=pd_ODAll['Coord_Y'].to_numpy()/(Lat-1)*(Ymax-Ymin)+Ymin
    
    
    _, idxV_All, LatModif, LongModif, LongMin, LongMax, LatMin, LatMax = InputDataZone(Feuillet, pd_ID, Long, Lat, Xmin, Xmax, Ymin, Ymax)
       
    nbsim=np.size(clf_all,axis=0)
    
    log_predicted=[]
    pred_all=[]
    for T in range(0,nbsim):
        pred=predicted[T]
        pred[pred==1]=(400-1)/400
        pred[pred==0]=1/400
        log_predicted.append( np.log( pred/(1-pred) ))
        pred_all.append( pred )
    
    return_pred= np.mean(log_predicted,axis=0)
    risk_pred= np.std(log_predicted, ddof=1,axis=0)

    return_pred1= np.mean(pred_all,axis=0)
    risk_pred1= np.std(pred_all, ddof=1,axis=0)
    
    Data=risk_pred.reshape(Lat,Long)
    plt.figure().set_figwidth(10)
    fig, ax = plt.subplots(figsize=(6,7), facecolor='w')
    plt.scatter(return_pred1[idxV_All],risk_pred1[idxV_All], color='cyan',marker="o", s=2,) 
    plt.plot(return_pred1[IsMineAllIdx.ravel()],risk_pred1[IsMineAllIdx.ravel()],"^w", markersize=10, mec = 'k') 
    plt.plot(return_pred1[IsMineIdx.ravel()],risk_pred1[IsMineIdx.ravel()],"^k", markersize=10, mec = 'k') 
    plt.plot([0,1],[0.1,0.1],"k") 
    plt.plot([0.5, 0.5],[0,0.5],"k") 
    ax.fill_between([0,0.5], 0, 0.1, alpha=0.25, color='orange')
    ax.fill_between([0,0.5], 0.1, 0.5, alpha=0.25, color='r')
    ax.fill_between([0.5,1], 0, 0.1, alpha=0.25, color='b')
    ax.fill_between([0.5,1], 0.1, 0.5, alpha=0.25, color='g')
    ax.set_xlabel('Return', fontsize=22)
    ax.set_ylabel('Risk', fontsize=22)
    ax.set_xlim(0,1)
    ax.set_ylim(0, 0.5)
    legend_elements = [Line2D([0], [0], color='r', lw=2, label='High-Risk/Low-Return'),
                       Line2D([0], [0], color='orange', lw=2, label='Low-Risk/Low-Return'),
                       Line2D([0], [0], color='g', lw=2, label='High-Risk/High-Return'),                       
                       Line2D([0], [0], color='b', lw=2, label='Low-Risk/High-Return'),]
    ax.set_box_aspect(1)
    ax.legend(handles=legend_elements, loc='lower right')
    plt.gca().invert_yaxis()
    plt.show()
    
    
    Data=risk_pred.reshape(Lat,Long)
    plt.figure().set_figwidth(10)
    fig, ax = plt.subplots(figsize=(6,7), facecolor='w')
    plt.scatter(return_pred[idxV_All],risk_pred[idxV_All], color='cyan',marker="o", s=2,) 
    plt.plot(return_pred[IsMineAllIdx.ravel()],risk_pred[IsMineAllIdx.ravel()],"^w", markersize=10, mec = 'k') 
    plt.plot(return_pred[IsMineIdx.ravel()],risk_pred[IsMineIdx.ravel()],"^k", markersize=10, mec = 'k') 
    plt.plot([-5.5,5.5],[0.5,0.5],"k") 
    plt.plot([0, 0],[0,2.5],"k") 
    ax.fill_between([-5.5,0], 0, 0.5, alpha=0.25, color='orange')
    ax.fill_between([-5.5,0], 0.5, 2.5, alpha=0.25, color='r')
    ax.fill_between([0,5.5], 0, 0.5, alpha=0.25, color='b')
    ax.fill_between([0,5.5], 0.5, 2.5, alpha=0.25, color='g')
    ax.set_xlabel('Return', fontsize=22)
    ax.set_ylabel('Risk', fontsize=22)
    ax.set_xlim(-5.5,5.5)
    ax.set_ylim(0, 2.5)
    legend_elements = [Line2D([0], [0], color='r', lw=2, label='High-Risk/Low-Return'),
                       Line2D([0], [0], color='orange', lw=2, label='Low-Risk/Low-Return'),
                       Line2D([0], [0], color='g', lw=2, label='High-Risk/High-Return'),                       
                       Line2D([0], [0], color='b', lw=2, label='Low-Risk/High-Return'),]
    ax.set_box_aspect(1)
    ax.legend(handles=legend_elements, loc='lower right')
    plt.gca().invert_yaxis()
    plt.show()
    
    
    Data=pMean.reshape(Lat,Long)
    plt.figure().set_figwidth(10)
    fig, ax = plt.subplots(figsize=(6,7),facecolor='w')
    im = ax.pcolormesh( LongData, LatData, Data, shading='auto', cmap='turbo', vmin=0., vmax=1)
    plt.plot(pd_ODAll[pd_ODAll['Mine']==1]['Coord_X'],pd_ODAll[pd_ODAll['Mine']==1]['Coord_Y'],"*w", markersize=14, mec = 'k') 
    plt.plot(pd_ODAll[pd_ODAll['Deposit']==1]['Coord_X'],pd_ODAll[pd_ODAll['Deposit']==1]['Coord_Y'],"ow", markersize=13, mec = 'k') 
    plt.plot(pd_ODAll[pd_ODAll['Occurence']==1]['Coord_X'],pd_ODAll[pd_ODAll['Occurence']==1]['Coord_Y'],"^w", markersize=12, mec = 'k') 
    plt.plot(pd_OD[pd_OD['Mine']==1]['Coord_X'],pd_OD[pd_OD['Mine']==1]['Coord_Y'],"*k", markersize=14) 
    plt.plot(pd_OD[pd_OD['Deposit']==1]['Coord_X'],pd_OD[pd_OD['Deposit']==1]['Coord_Y'],"ok", markersize=13) 
    plt.plot(pd_OD[pd_OD['Occurence']==1]['Coord_X'],pd_OD[pd_OD['Occurence']==1]['Coord_Y'],"^k", markersize=12) 
    plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.044, pad=0.1) 
    ax.set_xlabel('Longitude', fontsize=22)
    ax.set_ylabel('Latitude', fontsize=22)
    ax.set_xlim(LongMin, LongMax)
    ax.set_ylim(LatMin, LatMax)
    ax.set_box_aspect(1)
    plt.show() 
    
    Data=pStd.reshape(Lat,Long)
    plt.figure().set_figwidth(10)
    fig, ax = plt.subplots(figsize=(6,7), facecolor='w')
    im = ax.pcolormesh( LongData, LatData, Data, shading='auto', cmap='magma', vmin=0., vmax=0.10)
    plt.plot(pd_ODAll[pd_ODAll['Mine']==1]['Coord_X'],pd_ODAll[pd_ODAll['Mine']==1]['Coord_Y'],"*w", markersize=14, mec = 'k') 
    plt.plot(pd_ODAll[pd_ODAll['Deposit']==1]['Coord_X'],pd_ODAll[pd_ODAll['Deposit']==1]['Coord_Y'],"ow", markersize=13, mec = 'k') 
    plt.plot(pd_ODAll[pd_ODAll['Occurence']==1]['Coord_X'],pd_ODAll[pd_ODAll['Occurence']==1]['Coord_Y'],"^w", markersize=12, mec = 'k') 
    plt.plot(pd_OD[pd_OD['Mine']==1]['Coord_X'],pd_OD[pd_OD['Mine']==1]['Coord_Y'],"*k", markersize=14) 
    plt.plot(pd_OD[pd_OD['Deposit']==1]['Coord_X'],pd_OD[pd_OD['Deposit']==1]['Coord_Y'],"ok", markersize=13) 
    plt.plot(pd_OD[pd_OD['Occurence']==1]['Coord_X'],pd_OD[pd_OD['Occurence']==1]['Coord_Y'],"^k", markersize=12) 
    plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.044, pad=0.1) 
    ax.set_xlabel('Longitude', fontsize=22)
    ax.set_ylabel('Latitude', fontsize=22)
    ax.set_xlim(LongMin, LongMax)
    ax.set_ylim(LatMin, LatMax)
    ax.set_box_aspect(1)
    plt.show() 

    from matplotlib import colors
    cmap = colors.ListedColormap(['white','red'])  
    Data=pMean.reshape(Lat,Long)>0.5
    plt.figure().set_figwidth(10)
    fig, ax = plt.subplots(figsize=(6,7), facecolor='w')
    im = ax.pcolormesh( LongData, LatData, Data, shading='nearest', cmap=cmap, vmin=0., vmax=1)
    plt.plot(pd_ODAll[pd_ODAll['Mine']==1]['Coord_X'],pd_ODAll[pd_ODAll['Mine']==1]['Coord_Y'],"*w", markersize=14, mec = 'k') 
    plt.plot(pd_ODAll[pd_ODAll['Deposit']==1]['Coord_X'],pd_ODAll[pd_ODAll['Deposit']==1]['Coord_Y'],"ow", markersize=13, mec = 'k') 
    plt.plot(pd_ODAll[pd_ODAll['Occurence']==1]['Coord_X'],pd_ODAll[pd_ODAll['Occurence']==1]['Coord_Y'],"^w", markersize=12, mec = 'k') 
    plt.plot(pd_OD[pd_OD['Mine']==1]['Coord_X'],pd_OD[pd_OD['Mine']==1]['Coord_Y'],"*k", markersize=14) 
    plt.plot(pd_OD[pd_OD['Deposit']==1]['Coord_X'],pd_OD[pd_OD['Deposit']==1]['Coord_Y'],"ok", markersize=13) 
    plt.plot(pd_OD[pd_OD['Occurence']==1]['Coord_X'],pd_OD[pd_OD['Occurence']==1]['Coord_Y'],"^k", markersize=12) 
    plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.044, pad=0.1) 
    ax.set_xlabel('Longitude', fontsize=22)
    ax.set_ylabel('Latitude', fontsize=22)
    ax.set_xlim(LongMin, LongMax)
    ax.set_ylim(LatMin, LatMax)
    ax.set_box_aspect(1)
    plt.show() 
    
    print(np.mean(Data))
    
    cmap = colors.ListedColormap(['white','red'])  
    Data=pMean.reshape(Lat,Long)>np.quantile(pMean,0.90)
    plt.figure().set_figwidth(10)
    fig, ax = plt.subplots(figsize=(6,7), facecolor='w')
    im = ax.pcolormesh( LongData, LatData, Data, shading='nearest', cmap=cmap, vmin=0., vmax=1)
    plt.plot(pd_ODAll[pd_ODAll['Mine']==1]['Coord_X'],pd_ODAll[pd_ODAll['Mine']==1]['Coord_Y'],"*w", markersize=14, mec = 'k') 
    plt.plot(pd_ODAll[pd_ODAll['Deposit']==1]['Coord_X'],pd_ODAll[pd_ODAll['Deposit']==1]['Coord_Y'],"ow", markersize=13, mec = 'k') 
    plt.plot(pd_ODAll[pd_ODAll['Occurence']==1]['Coord_X'],pd_ODAll[pd_ODAll['Occurence']==1]['Coord_Y'],"^w", markersize=12, mec = 'k') 
    plt.plot(pd_OD[pd_OD['Mine']==1]['Coord_X'],pd_OD[pd_OD['Mine']==1]['Coord_Y'],"*k", markersize=14) 
    plt.plot(pd_OD[pd_OD['Deposit']==1]['Coord_X'],pd_OD[pd_OD['Deposit']==1]['Coord_Y'],"ok", markersize=13) 
    plt.plot(pd_OD[pd_OD['Occurence']==1]['Coord_X'],pd_OD[pd_OD['Occurence']==1]['Coord_Y'],"^k", markersize=12) 
    plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.044, pad=0.1) 
    ax.set_xlabel('Longitude', fontsize=22)
    ax.set_ylabel('Latitude', fontsize=22)
    ax.set_xlim(LongMin, LongMax)
    ax.set_ylim(LatMin, LatMax)
    ax.set_box_aspect(1)
    plt.show()
    
    
    seuil1=max(0.5,np.quantile(pMean,0.20))
    seuil2=max(0.5,np.quantile(pMean,0.40))
    seuil3=max(0.5,np.quantile(pMean,0.60))
    seuil4=max(0.5,np.quantile(pMean,0.75))
    seuil5=max(0.5,np.quantile(pMean,0.90))
    
    
    cmap = colors.ListedColormap(['white','green','lime', 'yellow','gold', 'red'])   
    Data=np.copy(pMean)
    Data[pMean<0.5]=0
    Data[pMean>seuil1]=1
    Data[pMean>seuil2]=2
    Data[pMean>seuil3]=3
    Data[pMean>seuil4]=4
    Data[pMean>seuil5]=5
    Data=Data.reshape(Lat,Long)
    
    plt.figure().set_figwidth(10)
    fig, ax = plt.subplots(figsize=(6,7), facecolor='w')
    im = ax.pcolormesh( LongData, LatData, Data, shading='nearest', cmap=cmap, vmin=0., vmax=5)
    plt.plot(pd_ODAll[pd_ODAll['Mine']==1]['Coord_X'],pd_ODAll[pd_ODAll['Mine']==1]['Coord_Y'],"*w", markersize=14, mec = 'k') 
    plt.plot(pd_ODAll[pd_ODAll['Deposit']==1]['Coord_X'],pd_ODAll[pd_ODAll['Deposit']==1]['Coord_Y'],"ow", markersize=13, mec = 'k') 
    plt.plot(pd_ODAll[pd_ODAll['Occurence']==1]['Coord_X'],pd_ODAll[pd_ODAll['Occurence']==1]['Coord_Y'],"^w", markersize=12, mec = 'k') 
    plt.plot(pd_OD[pd_OD['Mine']==1]['Coord_X'],pd_OD[pd_OD['Mine']==1]['Coord_Y'],"*k", markersize=14) 
    plt.plot(pd_OD[pd_OD['Deposit']==1]['Coord_X'],pd_OD[pd_OD['Deposit']==1]['Coord_Y'],"ok", markersize=13) 
    plt.plot(pd_OD[pd_OD['Occurence']==1]['Coord_X'],pd_OD[pd_OD['Occurence']==1]['Coord_Y'],"^k", markersize=12) 
    plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.044, pad=0.1) 
    ax.set_xlabel('Longitude', fontsize=22)
    ax.set_ylabel('Latitude', fontsize=22)
    ax.set_xlim(LongMin, LongMax)
    ax.set_ylim(LatMin, LatMax)
    ax.set_box_aspect(1)
    plt.show()
    
    seuil1=0.5
    seuil2=0.6
    seuil3=0.7
    seuil4=0.8
    seuil5=0.9
    
    from matplotlib import colors
    cmap = colors.ListedColormap(['white','green','lime', 'yellow','gold', 'red'])   
    Data=np.copy(pMean)
    Data[pMean<0.5]=0
    Data[pMean>seuil1]=1
    Data[pMean>seuil2]=2
    Data[pMean>seuil3]=3
    Data[pMean>seuil4]=4
    Data[pMean>seuil5]=5
    Data=Data.reshape(Lat,Long)
    
    plt.figure().set_figwidth(10)
    fig, ax = plt.subplots(figsize=(6,7), facecolor='w')
    im = ax.pcolormesh( LongData, LatData, Data, shading='nearest', cmap=cmap, vmin=0., vmax=5)
    plt.plot(pd_ODAll[pd_ODAll['Mine']==1]['Coord_X'],pd_ODAll[pd_ODAll['Mine']==1]['Coord_Y'],"*w", markersize=14, mec = 'k') 
    plt.plot(pd_ODAll[pd_ODAll['Deposit']==1]['Coord_X'],pd_ODAll[pd_ODAll['Deposit']==1]['Coord_Y'],"ow", markersize=13, mec = 'k') 
    plt.plot(pd_ODAll[pd_ODAll['Occurence']==1]['Coord_X'],pd_ODAll[pd_ODAll['Occurence']==1]['Coord_Y'],"^w", markersize=12, mec = 'k') 
    plt.plot(pd_OD[pd_OD['Mine']==1]['Coord_X'],pd_OD[pd_OD['Mine']==1]['Coord_Y'],"*k", markersize=14) 
    plt.plot(pd_OD[pd_OD['Deposit']==1]['Coord_X'],pd_OD[pd_OD['Deposit']==1]['Coord_Y'],"ok", markersize=13) 
    plt.plot(pd_OD[pd_OD['Occurence']==1]['Coord_X'],pd_OD[pd_OD['Occurence']==1]['Coord_Y'],"^k", markersize=12) 
    plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.044, pad=0.1) 
    ax.set_xlabel('Longitude', fontsize=22)
    ax.set_ylabel('Latitude', fontsize=22)
    ax.set_xlim(LongMin, LongMax)
    ax.set_ylim(LatMin, LatMax)
    ax.set_box_aspect(1)
    plt.show()
    
    Map=Data.ravel()
    B=Map    
    print([np.mean(B==0),np.mean(B==1),np.mean(B==2),np.mean(B==3),np.mean(B==4),np.mean(B==5)])   
    Map=Data.ravel()
    B=Map[IsMineIdx]     
    print([np.sum(B==0),np.sum(B==1),np.sum(B==2),np.sum(B==3),np.sum(B==4),np.sum(B==5)])
    Map=Data.ravel()   
    B=Map[IsMineAllIdx]     
    print([np.sum(B==0),np.sum(B==1),np.sum(B==2),np.sum(B==3),np.sum(B==4),np.sum(B==5)])
    
        
        
    print([ np.mean(pMean[IsMineIdx]), np.mean(pStd[IsMineIdx])]) 
    print([ np.mean(pMean[IsMineAllIdx]), np.mean(pStd[IsMineAllIdx])])   


    Title=pd_ID.columns[[x+2 for x in fea]].to_numpy().astype(str)
    Title[14]='R_AUI'
    shap_values=[]
    import shap   
    plt.figure()
    for T in range(0,nbsim):
        explainer = shap.TreeExplainer(clf_all[T])

        S_values = explainer.shap_values(X_Test[:,fea,T])
        shap_values.append( S_values )
    
    #SV1=[]
    #SV2=[]
    #X_Data=[]
    #for T in range(0,nbsim):
    #    if T==0:
    #        SV1=shap_values[T][0]
    #        SV2=shap_values[T][1]
    #        X_Data=X_Test[:,fea,T]
    #    else:
    #        SV1=np.concatenate((SV1,shap_values[T][0]),axis=0)
    #        SV2=np.concatenate((SV2,shap_values[T][1]),axis=0)
    #        X_Data=np.concatenate((X_Data,X_Test[:,fea,T]),axis=0)
    #    
    #plt.figure()
    #shap.summary_plot(SV2, X_Data, feature_names=Title,show=False,max_display=15)
    #plt.figure()
    #shap.summary_plot(SV2, X_Data, feature_names=Title,show=False,max_display=15, plot_type='bar')   

    tn=np.zeros((nbsim,1))
    fp=np.zeros((nbsim,1))
    fn=np.zeros((nbsim,1))
    tp=np.zeros((nbsim,1))
    thresholds=[]
    tpr=[]
    fpr=[]
    roc_auc=[]
    
    plt.figure()
    fig, ax = plt.subplots(figsize=(5,5), facecolor='w')
    legend_elements = [Line2D([0], [0], color='b', lw=4, label='Realization'),
                   Line2D([0], [0], color='r', lw=4, label='Average'),]
    
    for T in range(0,nbsim):
        fpr_T, tpr_T, thresholds_T = metrics.roc_curve(Y_Test[:,-1,T],  Proba_all[T][:,1])
        roc_auc_T = metrics.auc(fpr_T, tpr_T)
                
        f1 = interpolate.interp1d(thresholds_T, fpr_T,bounds_error=False)
        f2 = interpolate.interp1d(thresholds_T, tpr_T,bounds_error=False)
                
        thresholds_T=np.concatenate(([5],np.arange(1, 0, -0.01)))
        fpr_T = f1(thresholds_T)
        tpr_T = f2(thresholds_T)
        
        
        fpr.append(fpr_T)
        tpr.append(tpr_T)
        thresholds.append(thresholds_T)
        roc_auc.append(roc_auc_T)
                    
        plt.plot(fpr_T, tpr_T, 'b')
            
        tn[T], fp[T], fn[T], tp[T] = confusion_matrix(Y_Test[:,-1,T],Class_all[T]).ravel() 
    
    textstr = '\n'.join( (r'$\overline{AUC}$=%.2f' % (np.mean(roc_auc)),
                          r'$\sigma_{AUC}$=%.2f' % (np.std(roc_auc)),
           ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.plot(np.mean(fpr,axis=0), np.mean(tpr,axis=0), 'r', linewidth=3)
    ax.text(0.6, 0.45, textstr, fontsize=18,
        verticalalignment='top', bbox=props)
    
    plt.ylim([-0.01,1.01])
    plt.xlim([-0.01,1.01]) 
    ax.set_ylabel('True positive rate', fontsize=18)
    ax.set_xlabel('False positive rate', fontsize=18)
    ax.legend(handles=legend_elements, loc='lower right')
    plt.show
        
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(5,5), facecolor='w')

    cfms1=np.array([[np.mean(tn), np.mean(fp)],[np.mean(fn), np.mean(tp)]])
    sns.heatmap(cfms1, annot=True, ax=ax, yticklabels=['Negative', 'Positive'],
               xticklabels=['Negative', 'Positive'], annot_kws={"fontsize": 12}, vmin=0, vmax=np.sum(cfms1)//2, cmap='CMRmap', fmt='.1f')
    ax.set_ylabel('True label', fontsize=18)
    ax.set_xlabel('Predicted label', fontsize=18)
    plt.tight_layout()
    plt.show()
    
    
    
    
    
        
        
        

    
        
        
        
         