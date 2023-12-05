# -*- coding: utf-8 -*-
"""
Created on Wed May 31 21:22:42 2023

@author: dany-
"""

# Library importantion
import numpy as np

# %% Process  the inputdata from raw data
from InputData import InputDataProcess as IDP

ListOfParameters = ["MagHR", "MagGradHR","SigAnoMagHR",     # Geophys Haute resolution
                    "BouguerBR","AnoGrav1DVBR",             # Geophys Basse resolution
                    "Contact","Fault",                     # Geologie structure
                    "GeoRegio","GeoGen",                   ]# Geologie unité] 

Feuillet = 'F32F'

pd_ID = IDP(ListOfParameters, Feuillet)

from OutputData import OutputDataProcess as ODP

Mineral = 'Cu'
CorpsMine='Occurence'

pd_OD = ODP(Mineral, Feuillet, CorpsMine)

# %% Distance to mine 

from Functional import Dimension, Grid
from Functional import AdaptatedSpatialAssociation as ASA
Feuillet = 'F32J'

Xmin, Xmax, Ymin, Ymax, Lat, Long = Dimension(Feuillet) 
Grid = Grid(Long, Lat)

GeoASA = ASA(pd_OD, pd_ID, Feuillet)

# %%
from Functional import Dist2Mine as D2M
from Functional import Dist2Fault as D2F
from Functional import Dist2Contact as D2C
from Functional import ComputeDistanceToMineralization as D2Mine

DistMine=D2Mine(pd_ID, pd_OD)

dM=D2M(pd_OD)
dF=D2F(pd_ID,pd_OD)
dC=D2C(pd_ID,pd_OD)

# %% Building data set
from Functional import BuildData

seed= 151525
nbsim=100

TrainingData, ValidationData, TrainingLoc, ValidationLoc = BuildData(Feuillet, pd_ID, pd_OD, seed, nbsim)


    