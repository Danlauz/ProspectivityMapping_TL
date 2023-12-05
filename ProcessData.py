# -*- coding: utf-8 -*-

"""
Created on 2023-02-15
A Python program to process input data depending on the nature of the geological data.

This code is released from the paper of Lauzon et al. (2023) published in the journal XXX.

code author: Dany Lauzon
email: dany.lauzon@inrs.ca,
@author: Dany Lauzon

"""
import pandas as pd
import numpy as np
from numpy import save
from ProcessingDataFunction import InputDataProcess as IDP
from ProcessingDataFunction import OutputDataProcess as ODP
from ProcessingDataFunction import RegressionKrigingGeoRock
from ProcessingDataFunction import RegressionKrigingGeoSediment
from Functional import BuildData
    
# %% Process the RawData
Method = 'RF'
    
ListOfParameters = ["MagHR", "MagGradHR","SigAnoMagHR",     # Geophys Haute resolution
                    "BouguerBR","AnoGrav1DVBR",             # Geophys Basse resolution
                    "Contact","Fault",                     # Geologie structure
                    "GeoRegio","GeoGen",                   ]# Geologie unité] 

for Feuillet in ["F32J","F32F","F33FG","F32D","F32G"] :
    pd_ID = IDP(ListOfParameters, Feuillet)

    Element=["K2O","MGO","NA2O","CAO","FE2O3","MNO","SIO2","TIO2","AL2O3","P2O5",
              "AS_O","AU","AG","CU","SB","W","MO","NI","ZN","PB"]
    dp_RK_Rock = RegressionKrigingGeoRock(Feuillet,Element)
    
    Element=["AS_O","AU","AG","CU","HG","SB","W","MO","NI","ZN","MN","PB"]
    dp_RK_Sediment = RegressionKrigingGeoSediment(Feuillet,Element)
    
    df=pd.concat([pd_ID, dp_RK_Rock.iloc[:,2:], dp_RK_Sediment.iloc[:,2:]],axis=1)
    Path ="ProcessData\\InputData" + "\\" + Feuillet + ".csv"
    df.to_csv(Path, index=False) 

# %% Building Data for training and save them   

for Feuillet in ["F32J","F33FG","F32D","F32G","F32F"] :
    pd_OD = ODP('Cu_PS', Feuillet)
    pd_OD = ODP('Cu_P', Feuillet)

Mineral='Cu_P'
seed= 151525
nbsim=200

pd_ID=[]
pd_OD=[]
TrainingData=[]
ValidationData=[]
TrainingLoc=[]
ValidationLoc=[]
IdxTraining=[]
IdxValidation=[]
x=0
for Feuillet in ["F32J","F33FG","F32D","F32G","F32F"] :
    pd_ID.append(pd.read_csv("ProcessData\\InputData" + "\\" + Feuillet + ".csv"))
    pd_OD.append( pd.read_csv("ProcessData\\OccurenceData\\" + Feuillet + "_"+ Mineral +  ".csv"))

    TD, VD, TL, VL, IndT, IndV = BuildData(Feuillet, pd_ID[x], pd_OD[x], seed, nbsim)
    TrainingData.append(TD)
    ValidationData.append(VD)
    TrainingLoc.append(TL)
    ValidationLoc.append(VL)
    IdxTraining.append(IndT)
    IdxValidation.append(IndV)
    
    save("ProcessData\\TrainingData\\" + Feuillet + "_Data.npy", TD)
    save("ProcessData\\ValidationData\\" + Feuillet + "_Data.npy", VD)
    save("ProcessData\\TrainingData\\" + Feuillet + "_Loc.npy", TL)
    save("ProcessData\\ValidationData\\" + Feuillet + "_Loc.npy", VL)
    save("ProcessData\\IndicatorNegData\\" + Feuillet + "_Training.npy", IndT)
    save("ProcessData\\IndicatorNegData\\" + Feuillet + "_Val.npy", IndV)
    
    x=x+1
    
 
TrainingData_All=np.concatenate((TrainingData[0],TrainingData[1],TrainingData[2],TrainingData[3],TrainingData[4])) 
TrainingLoc_All=np.concatenate((TrainingLoc[0],TrainingLoc[1],TrainingLoc[2],TrainingLoc[3],TrainingLoc[4]))

save("ProcessData\\TrainingData\\DataAll.npy", TrainingData_All)
save("ProcessData\\ValidationData\\DataAll.npy", TrainingLoc_All)


Feuillet = "F32J"
            
TrainingData_All=np.concatenate((TrainingData[0],TrainingData[1],TrainingData[2],TrainingData[3],TrainingData[4],
                                 ValidationData[1],ValidationData[2],ValidationData[3],ValidationData[4])) 
TrainingLoc_All=np.concatenate((TrainingLoc[0],TrainingLoc[1],TrainingLoc[2],TrainingLoc[3],TrainingLoc[4],
                                ValidationLoc[1],ValidationLoc[2],ValidationLoc[3],ValidationLoc[4]))

save("ProcessData\\TrainingData\\" + Feuillet + "_DataAll.npy", TrainingData_All)
save("ProcessData\\ValidationData\\" + Feuillet + "_DataAll.npy", TrainingLoc_All)

Feuillet = "F33FG"
            
TrainingData_All=np.concatenate((TrainingData[0],TrainingData[1],TrainingData[2],TrainingData[3],TrainingData[4],
                                 ValidationData[0],ValidationData[2],ValidationData[3],ValidationData[4])) 
TrainingLoc_All=np.concatenate((TrainingLoc[0],TrainingLoc[1],TrainingLoc[2],TrainingLoc[3],TrainingLoc[4],
                                ValidationLoc[0],ValidationLoc[2],ValidationLoc[3],ValidationLoc[4]))

save("ProcessData\\TrainingData\\" + Feuillet + "_DataAll.npy", TrainingData_All)
save("ProcessData\\ValidationData\\" + Feuillet + "_DataAll.npy", TrainingLoc_All)

Feuillet = "F32D"
            
TrainingData_All=np.concatenate((TrainingData[0],TrainingData[1],TrainingData[2],TrainingData[3],TrainingData[4],
                                 ValidationData[0],ValidationData[1],ValidationData[3],ValidationData[4])) 
TrainingLoc_All=np.concatenate((TrainingLoc[0],TrainingLoc[1],TrainingLoc[2],TrainingLoc[3],TrainingLoc[4],
                                ValidationLoc[0],ValidationLoc[1],ValidationLoc[3],ValidationLoc[4]))

save("ProcessData\\TrainingData\\" + Feuillet + "_DataAll.npy", TrainingData_All)
save("ProcessData\\ValidationData\\" + Feuillet + "_DataAll.npy", TrainingLoc_All)

Feuillet = "F32G"
            
TrainingData_All=np.concatenate((TrainingData[0],TrainingData[1],TrainingData[2],TrainingData[3],TrainingData[4],
                                 ValidationData[0],ValidationData[1],ValidationData[2],ValidationData[4])) 
TrainingLoc_All=np.concatenate((TrainingLoc[0],TrainingLoc[1],TrainingLoc[2],TrainingLoc[3],TrainingLoc[4],
                                ValidationLoc[0],ValidationLoc[1],ValidationLoc[2],ValidationLoc[4]))

save("ProcessData\\TrainingData\\" + Feuillet + "_DataAll.npy", TrainingData_All)
save("ProcessData\\ValidationData\\" + Feuillet + "_DataAll.npy", TrainingLoc_All)

Feuillet = "F32F"
            
TrainingData_All=np.concatenate((TrainingData[0],TrainingData[1],TrainingData[2],TrainingData[3],TrainingData[4],
                                 ValidationData[0],ValidationData[1],ValidationData[2],ValidationData[3])) 
TrainingLoc_All=np.concatenate((TrainingLoc[0],TrainingLoc[1],TrainingLoc[2],TrainingLoc[3],TrainingLoc[4],
                                ValidationLoc[0],ValidationLoc[1],ValidationLoc[2],ValidationLoc[3]))

save("ProcessData\\TrainingData\\" + Feuillet + "_DataAll.npy", TrainingData_All)
save("ProcessData\\ValidationData\\" + Feuillet + "_DataAll.npy", TrainingLoc_All)



Feuillet = "F32J"
            
TrainingData_All=np.concatenate((TrainingData[1],TrainingData[2],TrainingData[3],TrainingData[4],
                                 ValidationData[1],ValidationData[2],ValidationData[3],ValidationData[4])) 
TrainingLoc_All=np.concatenate((TrainingLoc[1],TrainingLoc[2],TrainingLoc[3],TrainingLoc[4],
                                ValidationLoc[1],ValidationLoc[2],ValidationLoc[3],ValidationLoc[4]))

save("ProcessData\\TrainingData\\" + Feuillet + "_DataOther.npy", TrainingData_All)
save("ProcessData\\ValidationData\\" + Feuillet + "_DataOther.npy", TrainingLoc_All)

Feuillet = "F33FG"
            
TrainingData_All=np.concatenate((TrainingData[0],TrainingData[2],TrainingData[3],TrainingData[4],
                                 ValidationData[0],ValidationData[2],ValidationData[3],ValidationData[4])) 
TrainingLoc_All=np.concatenate((TrainingLoc[0],TrainingLoc[2],TrainingLoc[3],TrainingLoc[4],
                                ValidationLoc[0],ValidationLoc[2],ValidationLoc[3],ValidationLoc[4]))

save("ProcessData\\TrainingData\\" + Feuillet + "_DataOther.npy", TrainingData_All)
save("ProcessData\\ValidationData\\" + Feuillet + "_DataOther.npy", TrainingLoc_All)

Feuillet = "F32D"
            
TrainingData_All=np.concatenate((TrainingData[0],TrainingData[1],TrainingData[3],TrainingData[4],
                                 ValidationData[0],ValidationData[1],ValidationData[3],ValidationData[4])) 
TrainingLoc_All=np.concatenate((TrainingLoc[0],TrainingLoc[1],TrainingLoc[3],TrainingLoc[4],
                                ValidationLoc[0],ValidationLoc[1],ValidationLoc[3],ValidationLoc[4]))

save("ProcessData\\TrainingData\\" + Feuillet + "_DataOther.npy", TrainingData_All)
save("ProcessData\\ValidationData\\" + Feuillet + "_DataOther.npy", TrainingLoc_All)

Feuillet = "F32G"
            
TrainingData_All=np.concatenate((TrainingData[0],TrainingData[1],TrainingData[2],TrainingData[4],
                                 ValidationData[0],ValidationData[1],ValidationData[2],ValidationData[4])) 
TrainingLoc_All=np.concatenate((TrainingLoc[0],TrainingLoc[1],TrainingLoc[2],TrainingLoc[4],
                                ValidationLoc[0],ValidationLoc[1],ValidationLoc[2],ValidationLoc[4]))

save("ProcessData\\TrainingData\\" + Feuillet + "_DataOther.npy", TrainingData_All)
save("ProcessData\\ValidationData\\" + Feuillet + "_DataOther.npy", TrainingLoc_All)

Feuillet = "F32F"
            
TrainingData_All=np.concatenate((TrainingData[0],TrainingData[1],TrainingData[2],TrainingData[3],
                                 ValidationData[0],ValidationData[1],ValidationData[2],ValidationData[3])) 
TrainingLoc_All=np.concatenate((TrainingLoc[0],TrainingLoc[1],TrainingLoc[2],TrainingLoc[3],
                                ValidationLoc[0],ValidationLoc[1],ValidationLoc[2],ValidationLoc[3]))

save("ProcessData\\TrainingData\\" + Feuillet + "_DataOther.npy", TrainingData_All)
save("ProcessData\\ValidationData\\" + Feuillet + "_DataOther.npy", TrainingLoc_All)
