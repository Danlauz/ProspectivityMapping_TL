# -*- coding: utf-8 -*-
"""
Created on Wed May 31 22:41:47 2023

@author: dany-
"""
import numpy as np
from Functional import Dimension, SamplingNegativeData, BuildingNbSimDataSet
 
def BuildData(Feuillet, pd_ID, pd_OD, seed, nbsim):
           
    from Functional import Dist2Mine as D2M
    from Functional import Dist2Fault as D2F
    from Functional import Dist2Contact as D2C
    
    from Functional import AdaptatedSpatialAssociation as ASA
    from Functional import ComputeDistanceToMineralization as D2Mine

    ASA = ASA(pd_OD, pd_ID, Feuillet) 
    DistMine=D2Mine(pd_ID, pd_OD)
    
    
    Seuil=[D2M(pd_OD),D2F(pd_ID,pd_OD),D2C(pd_ID,pd_OD)]    
    # Sampling negative Data   
    idx_P1, idx_N1, CP1, CN1 = SamplingNegativeData(idx1, idx1_N, Feuillet, Seuil)
    TrainingData, TrainingLoc = BuildingNbSimDataSet(idx_P1, idx_N1, CP1, CN1)
     
    idx_P2, idx_N2, CP2, CN2 = SamplingNegativeData(idx2, idx2_N, Feuillet, Seuil)
    ValidationData, ValidationLoc = BuildingNbSimDataSet(idx_P2, idx_N2, CP2, CN2)
    
    return  TrainingData, ValidationData, TrainingLoc, ValidationLoc