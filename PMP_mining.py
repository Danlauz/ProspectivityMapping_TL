# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 13:26:52 2023

@author: dany-
"""

# -*- coding: utf-8 -*-

"""
Created on 2023-02-15
A Python program to read and process tiff images depending on the nature of the geological data.

This code is released from the paper of Lauzon et al. (2023) published in the journal XXX.

code author: Dany Lauzon
email: dany.lauzon@inrs.ca,
@author: Dany Lauzon

"""

import numpy as np
import random

# Processing the input data (geological information)
from ProcessData import InputDataProcess as ID
# Processing the output data (mineralization information)
from ProcessData import OutputDataProcess as OD
from ProcessData import ComputeDistanceToMineralization as distMin
from ProcessData import AdaptatedSpatialAssociation as ASA
from ProcessData import BuildData
from ProcessData import Dimension as DimFeuillet

# function for the classic classification with sample negative data 
from Classifier_PMP import Classifier_PMP as C_PMP

def ValidationZone(Feuillet, InputData, Long, Lat, Xmin, Xmax, Ymin, Ymax):
    if Feuillet == "F32D":
        idx1=InputData[:,0]>round(1*Long/2)
        idx2=InputData[:,0]<=round(1*Long/2)
        Long=int(round(1*Long/2)) 
        Xmax=1*(Xmax-Xmin)/2+Xmin             
    if Feuillet == "F33FG":
        idx1=InputData[:,0]>Long/2
        idx2=InputData[:,0]<=Long/2 
        Long=int(Long/2) 
        Xmax=(Xmax-Xmin)/2+Xmin          
    if Feuillet == "F32J":
        idx1=InputData[:,1]>Lat/2
        idx2=InputData[:,1]<=Lat/2
        Lat=int(Lat/2)
        Ymax=(Ymax-Ymin)/2+Ymin 
    if Feuillet == "F32F":
        idx1=InputData[:,0]>Long/2
        idx2=InputData[:,0]<=Long/2 
        Long=int(Long/2) 
        Xmax=(Xmax-Xmin)/2+Xmin   
    if Feuillet == "F32GAll":
        idx1=InputData[:,0]>round(1*Long/2)
        idx2=InputData[:,0]<=round(1*Long/2) 
        Long=int(round(1*Long/2)) 
        Xmax=1*(Xmax-Xmin)/2+Xmin          
    return idx1, idx2, Lat, Long, Xmin, Xmax, Ymin, Ymax 


def TrainingZone(IA, FeuilletAll, nbsim, ListOfParameters, Mineral, element, Title, CorpsMine , Figure, separate, count): 

    PN_Data_Training=[]
    PN_Data_Val=[]
    seed=1425
    for Feuillet in FeuilletAll: 
        Xmin, Xmax, Ymin, Ymax, Lat, Long = DimFeuillet(Feuillet) 
        
    # Process the input and output data for training
        seed=seed+10

        InputData = ID(ListOfParameters, Feuillet)
        OutputData, IsMineAll = OD(Mineral, Feuillet, CorpsMine)

        Grid=InputData[:,0:2] 
        DistMine=distMin(OutputData,Grid)

        Geo=InputData[:,13]
        Geo=Geo.reshape(np.size(Geo),1)
        DistGeo=ASA(OutputData, Geo, Feuillet) 
                 
        # Split data for training and validation
        PN_Data_T, PN_Data_V ,LocData_T, LocData_V = BuildData(OutputData, InputData, DistGeo, DistMine, element, Feuillet, seed, nbsim) 
        
        
        if np.size(PN_Data_Training)==0 :
            PN_Data_Training = PN_Data_T
            LocData_Training = LocData_T
        else: 
            PN_Data_Training = np.concatenate((PN_Data_Training, PN_Data_T),axis=0) 
            LocData_Training = np.concatenate((LocData_Training, LocData_T),axis=0) 
            
        if count==-1: 
            if np.size(PN_Data_Val)==0:
                PN_Data_Val = PN_Data_V
                LocData_Val = LocData_V
            else: 
                PN_Data_Val = np.concatenate((PN_Data_Val, PN_Data_V),axis=0) 
                LocData_Val = np.concatenate((LocData_Val, LocData_V),axis=0)
        elif count>=0:
            if FeuilletAll[count]==Feuillet:
                PN_Data_Val = PN_Data_V
                LocData_Val = LocData_V
                nbData=np.size(PN_Data_T,axis=0)
    
    # Initialize data for testing in another region
    LocT=LocData_Training[:, -1,0].argsort()  
    PN_Data_Training=PN_Data_Training[LocT,:,:]
    LocData_Training=LocData_Training[LocT,:,:]   
    
    LocV=LocData_Val[:, -1, 0 ].argsort()
    PN_Data_Val=PN_Data_Val[LocV,:,:]
    LocData_Val=LocData_Val[LocV,:,:]   
         
 
    clf_all=list()
    for T in range(0, nbsim):
        # RandomNegativeDat_Classifier_PMP 
        if separate==True:
            idxNT=random.sample(range(0, int(np.size(PN_Data_Training,axis=0)/2)), int(nbData/2))
            idxPT=random.sample(range(int(np.size(PN_Data_Training,axis=0)/2), np.size(PN_Data_Training,axis=0)), int(nbData/2) )
            idxT=np.append(idxNT,idxPT)
            clf, accuracy, ProbaTest, ClassTest = C_PMP(PN_Data_Training[idxT,:,T],PN_Data_Val[:,:,T], IA,  seed)                      
        else:

            clf, accuracy, ProbaTest, ClassTest = C_PMP(PN_Data_Training[:,:,T],PN_Data_Val[:,:,T], IA , seed)    
                     
        clf_all.append(clf)


       
    return clf_all, PN_Data_Training, PN_Data_Val, LocData_Training, LocData_Val, IsMineAll