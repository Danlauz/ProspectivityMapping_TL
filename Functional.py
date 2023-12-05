# -*- coding: utf-8 -*-
"""
Created on Wed May 31 21:09:18 2023

@author: dany-
"""

import numpy as np
from scipy.spatial import distance
from ECDF import ecdf
from scipy.stats import norm
from scipy.interpolate import interp1d

def Dimension(Feuillet):  
         
    if Feuillet == "F32D":
        Xmin, Xmax, Ymin, Ymax, Lat, Long= -79.5,-78.5, 48, 48.5, 250, 500

    if Feuillet == "F33FG":
        Xmin, Xmax, Ymin, Ymax, Lat, Long=-76.5, -75.5, 53.5, 53.75, 125, 500
        
    if Feuillet == "F32J":
        Xmin, Xmax, Ymin, Ymax, Lat, Long= -75, -74.5, 50.5, 51, 250, 250
        
    if Feuillet == "F32F":
        Xmin, Xmax, Ymin, Ymax, Lat, Long= -78, -76, 49, 50, 500, 1000
            
    if Feuillet == "F32G":
        Xmin, Xmax, Ymin, Ymax, Lat, Long= -76, -74, 49, 50, 500, 1000
        
    return Xmin, Xmax, Ymin, Ymax, int(Lat) , int(Long) 

def InputDataZone(Feuillet, pd_ID, Long, Lat, Xmin, Xmax, Ymin, Ymax):
    if Feuillet == "F32D":
        idx_T=pd_ID["Coord_X"]>round(2*Long/5)
        idx_V=pd_ID["Coord_X"]<=round(2*Long/5)
        Long=int(round(2*Long/5)) 
        Xmax=2*(Xmax-Xmin)/5+Xmin             
    if Feuillet == "F33FG":
        idx_T=pd_ID["Coord_X"]>Long/2
        idx_V=pd_ID["Coord_X"]<=Long/2 
        Long=int(Long/2) 
        Xmax=(Xmax-Xmin)/2+Xmin          
    if Feuillet == "F32J":
        idx_T=pd_ID["Coord_Y"]>Lat/2
        idx_V=pd_ID["Coord_Y"]<=Lat/2 
        Lat=int(Lat/2) 
        Ymax=(Ymax-Ymin)/2+Ymin 
    if Feuillet == "F32F":
        idx_T=pd_ID["Coord_X"]>Long/2
        idx_V=pd_ID["Coord_X"]<=Long/2 
        Long=int(Long/2) 
        Xmax=(Xmax-Xmin)/2+Xmin   
    if Feuillet == "F32G":
        idx_T=pd_ID["Coord_X"]>round(3.5*Long/5)
        idx_V=pd_ID["Coord_X"]<=round(3.5*Long/5) 
        Long=int(round(3.5*Long/5)) 
        Xmax=3.5*(Xmax-Xmin)/5+Xmin   
       
    return idx_T, idx_V, Lat, Long, Xmin, Xmax, Ymin, Ymax 

def OutputDataZone(Feuillet, pd_OD, Long, Lat):
    if Feuillet == "F32D":
        idx_T=pd_OD["Coord_X"]>round(2*Long/5)
        idx_V=pd_OD["Coord_X"]<=round(2*Long/5)
           
    if Feuillet == "F33FG":
        idx_T=pd_OD["Coord_X"]>Long/2
        idx_V=pd_OD["Coord_X"]<=Long/2 
         
    if Feuillet == "F32J":
        idx_T=pd_OD["Coord_Y"]>Lat/2
        idx_V=pd_OD["Coord_Y"]<=Lat/2 

    if Feuillet == "F32F":
        idx_T=pd_OD["Coord_X"]>Long/2
        idx_V=pd_OD["Coord_X"]<=Long/2 
  
    if Feuillet == "F32G":
        idx_T=pd_OD["Coord_X"]>round(3.5*Long/5)
        idx_V=pd_OD["Coord_X"]<=round(3.5*Long/5) 
         
    return idx_T, idx_V

def ComputeDistanceToMineralization(pd_ID, pd_OD):
    
    pd_Mine=pd_OD.copy()
    Grid_Mine=pd_Mine[["Coord_X","Coord_Y"]].to_numpy() 

    Grid=pd_ID[["Coord_X","Coord_Y"]].to_numpy() 
    
    Dist=distance.cdist(Grid, Grid_Mine, 'euclidean').min(axis=1)
    return Dist

def AdaptatedSpatialAssociation(pd_OD, pd_ID, Feuillet):
    Xmin, Xmax, Ymin, Ymax, Lat, Long = Dimension(Feuillet) 
    
    pd_Mine=pd_OD.copy()

    CoordX=pd_Mine['Coord_X'].to_numpy()
    CoordY=pd_Mine['Coord_Y'].to_numpy()

            
    GeoRegio= pd_ID['GeoRegio'].to_numpy()
    ListGeo= np.unique(GeoRegio) 
        
    Index= (Lat-CoordY-1)*Long + CoordX
    Index=(np.rint(Index)).astype(int)

    GeoASA=GeoRegio*10000

    N_A=Lat*Long
    N_D=np.sum(pd_OD["Index"])
    
    for GeoIndex in ListGeo:
        N_L= np.sum(GeoRegio==GeoIndex)
        N_LD= np.sum(GeoRegio[Index]==GeoIndex)
        ASA= (N_LD/N_D)/(N_L/N_A)
        GeoASA[GeoRegio==GeoIndex]=ASA
       
    return GeoASA

def Dist2Mine(pd_OD, Feuillet):
    if Feuillet=="F32D":
        Q=0.95
    else:
        Q=0.95
        
    pd_Mine=pd_OD.copy()            
    Grid=pd_Mine[["Coord_X","Coord_Y"]].to_numpy()    
    Dist=np.sort(distance.cdist(Grid, Grid, 'euclidean'),axis=1)[:,1]
        
    qe, pe = ecdf(Dist)
    qe=np.concatenate(([0],qe))
    pe=np.concatenate(([0],pe))
    dM=qe[pe<Q].max()
    import matplotlib.pyplot as plt    
    plt.figure().set_figwidth(10)
    plt.plot(qe,pe)                            
   
    return dM
    
def Dist2Fault(pd_ID,pd_OD):
 
    Dist=[]
    
    pd_Mine = pd_OD[(pd_OD["Index"]==1 )] 
    Grid=pd_Mine[["Coord_X","Coord_Y"]].to_numpy()  
    
    pd_F = pd_ID[(pd_ID["Fault_Binairy"]==1)] 
    GridF=pd_F[["Coord_X","Coord_Y"]].to_numpy() 
    
    if np.size(Dist)==0:        
        Dist=distance.cdist(Grid, GridF, 'euclidean').min(axis=1)
    else:
        Dist=np.append(Dist,distance.cdist(Grid, GridF, 'euclidean').min(axis=1))
                
    qe, pe = ecdf(Dist)
    qe=np.concatenate(([0],qe))
    pe=np.concatenate(([0],pe))
    dF=qe[pe<0.95].max()
       
    return dF 
 
def Dist2Contact(pd_ID,pd_OD):
       
    Dist=[]
    
    pd_Mine = pd_OD[(pd_OD["Index"]==1 )] 
    Grid=pd_Mine[["Coord_X","Coord_Y"]].to_numpy()  
    
    pd_C = pd_ID[(pd_ID["Contact_Binairy"]==1)] 
    GridC=pd_C[["Coord_X","Coord_Y"]].to_numpy() 
    
    if np.size(Dist)==0:        
        Dist=distance.cdist(Grid, GridC, 'euclidean').min(axis=1)
    else:
        Dist=np.append(Dist,distance.cdist(Grid, GridC, 'euclidean').min(axis=1))
              
    qe, pe = ecdf(Dist)
    qe=np.concatenate(([0],qe))
    pe=np.concatenate(([0],pe))
    dContact=qe[pe<0.95].max()
    
    return dContact 

def Grid(Long, Lat):
    nx, ny = (Long, Lat)
    xv, yv = np.meshgrid(np.linspace(0, nx-1, nx), np.linspace(ny-1, 0, ny))

    xv=xv.reshape(-1,1)
    yv=yv.reshape(-1,1)

    Grid=np.concatenate((xv, yv), axis=1)
    
    return Grid

def SamplingNegativeData(Feuillet, pd_ID, pd_OD, ASA, DistMine, seed, nbsim, idxMine, idxField, Seuil):
    
       
    Xmin, Xmax, Ymin, Ymax, Lat, Long = Dimension(Feuillet) 
    
    InputData= pd_ID.to_numpy()
    OutputData= pd_OD.to_numpy()  

    DataField=InputData[idxField,0:2]
    CP = OutputData[idxMine,2].sum()
    idx_P= idxMine & (OutputData[:,2]==1) 
    idx_P= (Lat-OutputData[idx_P,1]-1)*Long + OutputData[idx_P,0]
    idx_P=(np.rint(idx_P)).astype(int)
     
    idx_N=[]
    DistGeo=(ASA<=np.quantile(ASA,0.25)).reshape(-1,1)*1
    #np.reshape(InputData[idxField,9]>Seuil[2],(-1,1))*np.reshape(InputData[idxField,12]>Seuil[1],(-1,1))*(DistGeoMinRegio[idxField])*
    Indicator=(DistGeo[idxField])*np.reshape((DistMine[idxField]>Seuil[0]),(-1,1))+0 
    Indicator=np.reshape(Indicator/Indicator.sum(),(-1))

    np.random.seed(seed*4+2)
    for T in range(0,nbsim):
        idxN = np.random.choice(np.size(ASA[idxField]), int(CP) , p=Indicator, replace=False)
        if np.size(idx_N)==0:
            idxN= (Lat-DataField[idxN,1]-1)*Long + DataField[idxN,0]
            idxN=(np.rint(idxN)).astype(int)
            idx_N=np.reshape(idxN,(-1,1))
        else:
            idxN= (Lat-DataField[idxN,1]-1)*Long + DataField[idxN,0]
            idxN=(np.rint(idxN)).astype(int)
            idxN=np.reshape(idxN,(-1,1))
            idx_N=np.concatenate((idx_N,idxN),axis=1)
    CN=CP
    return idx_P, idx_N, CP, CN, Indicator
 
def BuildingNbSimDataSet(pd_ID, nbsim, idx_P, idx_N, CP, CN):
    
    InputData= pd_ID.to_numpy()
    
    TrainData=[]; 
    # Building the data for the training

    P_Data=np.concatenate( (InputData[idx_P,:],np.ones((int(CP),1))), axis=1)
     
    for T in range(0,nbsim):
        a=(np.rint(idx_N[:,T])).astype(int)
        N_Data=np.concatenate( (InputData[a,:],np.zeros((int(CN),1))), axis=1)
        PN_Data=np.concatenate((P_Data,N_Data),axis=0)    
        if np.size(TrainData)==0:
            TrainData=np.reshape(PN_Data,(np.size(PN_Data,0),np.size(PN_Data,1),1))
        else:
            PN_Data=np.reshape(PN_Data,(np.size(PN_Data,0),np.size(PN_Data,1),1))
            TrainData=np.concatenate((TrainData,PN_Data),axis=2)   
     
    LocData=TrainData[:,[0,1,-1],:]
    PN_Data=TrainData[:,2:-1,:]
     
    return PN_Data, LocData

def BuildData(Feuillet, pd_ID, pd_OD, seed, nbsim):
    Xmin, Xmax, Ymin, Ymax, Lat, Long = Dimension(Feuillet) 
      
    ASA = AdaptatedSpatialAssociation(pd_OD, pd_ID, Feuillet) 
    DistMine=ComputeDistanceToMineralization(pd_ID, pd_OD)
                                             
                                             
    idxT_OD, idxV_OD = OutputDataZone(Feuillet, pd_OD, Long, Lat)
    idxT_ID, idxV_ID, _, _, _, _, _, _ = InputDataZone(Feuillet, pd_ID, Long, Lat, Xmin, Xmax, Ymin, Ymax)
    
    Seuil=[Dist2Mine(pd_OD,Feuillet),Dist2Fault(pd_ID,pd_OD),Dist2Contact(pd_ID,pd_OD)]    
    # Sampling negative Data  

    idx_P1, idx_N1, CP1, CN1, IndicatorTraining = SamplingNegativeData(Feuillet, pd_ID, pd_OD, ASA, DistMine, seed, nbsim, idxT_OD, idxT_ID, Seuil)
    TrainingData, TrainingLoc = BuildingNbSimDataSet(pd_ID, nbsim, idx_P1, idx_N1, CP1, CN1)
     
    idx_P2, idx_N2, CP2, CN2, IndicatorValidation = SamplingNegativeData(Feuillet, pd_ID, pd_OD, ASA, DistMine, seed, nbsim, idxV_OD, idxV_ID, Seuil)
    ValidationData, ValidationLoc  = BuildingNbSimDataSet(pd_ID, nbsim, idx_P2, idx_N2, CP2, CN2)
    
    return  TrainingData, ValidationData, TrainingLoc, ValidationLoc, IndicatorTraining , IndicatorValidation 

def anamor(y):
    """
    Transform data into a Gaussian distribution.

    Parameters:
    -----------
    y: ndarray
        An n x (nc+1) matrix where nc is the number of coordinates (nc>=1).

    Returns:
    --------
    z: ndarray
        Transformed data in a Gaussian distribution.
    """
    n, p = y.shape
    n2 = len(np.unique(y[:, -1]))

    if n2 <= n:  # there are one or more equal values. We use a Gaussian white noise with low variance to break the equalities
        ys = y[:,-1] + np.random.randn(n)/10000
        ys = ys.ravel()

    else:
        ys = np.zeros((n, 1)).ravel()

    id1 = np.lexsort((ys, y[:, -1]))
    id2 = np.argsort(id1)
    rang = np.arange(1, n+1)
    z = norm.ppf(rang/(n+1), 0, 1)
    
    z=z[id2]
    z=z.reshape(len(z),1)
       
    zall = np.hstack((y, z))
    
    return z, zall

def anamorinv(y, w):
    """
    Transform Gaussian data back to original values.

    Parameters:
    -----------
    y: ndarray
        An n x (nc+2) matrix with the following entries: [coordinates, value, Gaussian value].
    w: ndarray
        Gaussian data simulated (ns x 1).

    Returns:
    --------
    z: ndarray
        Transformed data back to original values (ns x 2), [Gaussian value, original value].
    """
    n, p = y.shape
    yminv = 0  # minimum true value (change as needed)
    yming = -3.5  # minimum Gaussian value (change as needed)
    ymaxv = np.max(y[:, p-2])*1.2  # maximum true value (change as needed)
    ymaxg = 3.5  # maximum Gaussian value

    yt = y[:, [p-2, p-1]].copy()
    yt = yt[yt[:, 1].argsort()]
    y = np.vstack((np.array([yminv, yming]), yt, np.array([ymaxv, ymaxg])))

    f = interp1d(y[:, 1], y[:, 0], kind='linear', bounds_error=False, fill_value=(yminv, ymaxv))
    z = f(w)

    if p > 2:
        w=w.reshape(len(w),1)
        z=z.reshape(len(z),1)
        zall = np.hstack((w, z))

    return z