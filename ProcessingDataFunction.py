# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 00:14:11 2023

@author: dany-
"""
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from Functional import Dimension, anamorinv, anamor

from PIL import Image
from scipy.ndimage import uniform_filter
import dist_min as DistMinTree
import idw

from Classifier_PMP_ParamValid import Classifier_PMP_Tuning

def InputDataProcess(ListOfParameters, Feuillet):    
    print(Feuillet)   
     
    def LoadImage(Path):    
        im=Image.open(Path)
        return im

    def rgb2gray(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2126 * r + 0.7152 * g + 0.0722 * b    
        return gray

    def ResizeImage(im,Lat,Long,Param):
        if  Param == "GeoGen" or Param == "GeoRegio":
            im=im.resize( (Long,Lat), Image.NEAREST)
        else:
            im=im.resize( (Long,Lat), Image.BICUBIC)        
        im = list(im.getdata())
        length=np.size(im[0])
        im=np.array(im)
        im=im.reshape(Long,Lat,length)
        return im

    def Matrix2Vector(im):    
        imVector = np.reshape(im,(-1,1))
        return imVector
    
    def AppendVector(Vall,V):        
        Vall = np.append(Vall,V, axis=1)
        return Vall

    Xmin, Xmax, Ymin, Ymax, Lat, Long = Dimension(Feuillet)  
    xv, yv = np.meshgrid(np.linspace(0, Long-1, Long), np.linspace(Lat-1, 0, Lat))

    xv=xv.reshape(-1,1)
    yv=yv.reshape(-1,1)

    ImageV=np.concatenate((xv, yv), axis=1)
    Title=["Coord_X","Coord_Y"]

    for Param in ListOfParameters:
        
        if Param == "Contact" or Param == "Fault" or Param == "GeoGen":
            Format = ".jpg"
        else: 
            Format = ".tif"
            
        Path ="RawData\\" + Param + "\\" + Feuillet + Format
    
        ImageRGB = LoadImage(Path)
        
        ImageRGB = ResizeImage(ImageRGB,Lat, Long, Param)

        if np.size(ImageRGB,axis=2)>=3:
            ImageGray = rgb2gray(ImageRGB)/255
        else:
            ImageGray = ImageRGB
        
        ImV = Matrix2Vector(ImageGray)
           
        if Param == "Contact" or Param == "Fault":
            ImV = abs(ImV-1)  # reverse white-black color          
            ImV[ImV <= 0.005] = 0
            ImV[ImV > 0.005] = 1
            ImageV = AppendVector(ImageV,ImV)
                               
            ImV1=uniform_filter(ImV.reshape(Lat,Long), size=10, mode='constant').reshape(-1,1)
            ImageV = AppendVector(ImageV,ImV1)
                
            ImV = np.reshape(ImV,(-1))
            ImV=ImV/ImV.max()
            GridsOnes=ImageV[ImV==1,0:2]
            Gridx=ImageV[:,0:2]
                
            DistMin_tree = DistMinTree.tree(GridsOnes)
            ImV = DistMin_tree(Gridx).reshape(-1,1)

            ImV = np.reshape(ImV,(-1,1))
                               
            ImageV = AppendVector(ImageV,ImV)
            if Param == "Contact" :
                Title.append("Contact_Binairy") 
                Title.append("Contact_Density") 
                Title.append("Contact_Distance") 
            if Param == "Fault" :
                Title.append("Fault_Binairy") 
                Title.append("Fault_Density") 
                Title.append("Fault_Distance") 
                
        elif  Param == "GeoGen" :
            ImV=np.round(ImV / 0.01) * 0.01
                
            from collections import Counter

            def most_common(lst):
                counter = Counter(lst)
                most_common_values = counter.most_common()  # Returns a list of tuples [(value, count), ...]
                max_count = most_common_values[0][1]  # Get the count of the most common value
                most_common = [value for value, count in most_common_values if count == max_count]
                return most_common
                
            ImV=ImV.reshape(Lat,Long)
            ImV2=ImV.reshape(Lat,Long)
            for i in range(0,Long):
                for j in range(0,Lat):
                    Datalist=list(ImV[max(0,j-2):min(j+3,Lat),max(0,i-2):min(i+3,Long)].ravel())
                    if np.sum(most_common(Datalist)==ImV[j,i])==0:
                        ImV2[j,i]=most_common(Datalist)[0]
            ImV2=ImV.reshape(-1,1)          
            ImageV = AppendVector(ImageV,ImV2) 
            Title.append(Param)                  
        else:
            ImageV = AppendVector(ImageV,ImV)
            Title.append(Param) 
                
    Path ="ProcessData\\GeoStruct\\" + Feuillet + ".csv"
    print(Path)
    df = pd.DataFrame(ImageV, columns = Title)  
    df.to_csv(Path, index=False)
        
    return df
# %%
def OutputDataProcess(Mineral, Feuillet):
    
# Define Dimension 
        
    Xmin, Xmax, Ymin, Ymax, Lat, Long = Dimension(Feuillet) 

    Path ="RawData\\Occurence\\" + Feuillet + ".csv"
    TableData = pd.read_csv(Path)
        
    Data=TableData[TableData[Mineral]==1][["Coord_X","Coord_Y", Mineral, "Mine", "Deposit","Occurrence"]]    
    Data["Coord_X"]=np.around((Data["Coord_X"]-Xmin)/(Xmax-Xmin)*(Long-1))
    Data["Coord_Y"]=np.around((Data["Coord_Y"]-Ymin)/(Ymax-Ymin)*(Lat-1))

    Idx =  ((Lat-Data["Coord_Y"]-1)*Long+Data["Coord_X"]).to_numpy()
    IdxNew, ia, ic= np.unique(Idx,return_index=True,return_inverse=True, axis=0)
    
    DataNew=Data.iloc[ia,:].to_numpy()
        
    Title=["Coord_X","Coord_Y", "Index","Mine","Deposit","Occurence"]
    
    Path ="ProcessData\\OccurenceData\\" + Feuillet + "_"+ Mineral +  ".csv"
    print(Path)
    df = pd.DataFrame(DataNew, columns = Title)  
    df.to_csv(Path, index=False)    
    return  df
# %%
def RegressionKrigingGeoSediment(Feuillet, Element):

    models = RandomForestRegressor(n_estimators=400, random_state=85465, bootstrap=True, max_depth=None,
                                   min_samples_split=2, min_samples_leaf=1, max_features = 'auto')

    title=["Coord_X","Coord_Y","S_AS","S_AU","S_AG","S_CU","S_HG","S_SB","S_W","S_MO","S_NI","S_ZN","S_MN","S_PB"]

    Element_ID= [ "Coord_X", "Coord_Y",
	        "MagHR", "MagGradHR", "SigAnoMagHR", "BouguerBR", "AnoGrav1DVBR",
            "Contact_Binairy", "Contact_Density", "Contact_Distance",
            "Fault_Binairy", "Fault_Density",  "Fault_Distance", 
	        "GeoRegio"]

    print(Feuillet)
    # InputData for regression Kriging
    Path ="ProcessData\\GeoStruct\\" + Feuillet + ".csv"
    df = pd.read_csv(Path)
    InputData =  df[Element_ID].to_numpy()
    EV = InputData[:, 2:]
    # Define Domain Dimension 
    Xmin, Xmax, Ymin, Ymax, Lat, Long=Dimension(Feuillet)
    
    # Grid dimension
    xv, yv = np.meshgrid(np.linspace(Xmin, Xmax, Long),np.linspace(Ymax, Ymin, Lat) ,)
    xv=xv.reshape(-1,1)
    yv=yv.reshape(-1,1)
    Grid=np.concatenate((xv, yv), axis=1)

    # Tranform data to the original space
    interpolationMap=InputData[:,0:2]

    # Loading geochimical Data   
    Path ="RawData\\SedimentGeochemistry" + "\\" + Feuillet + ".csv"
    TableData = pd.read_csv(Path)

    # interpolation of each component
    for chim in Element:
        print(chim)
        # Selection of the element
        Data=TableData[["Coord_X","Coord_Y",chim]]
        Data=pd.DataFrame(Data).to_numpy()
        Data=Data[Data[:,-1]>0 ,:]

                   
        Grid_Data=np.concatenate((Data[:, 0].reshape(-1,1),Data[:, 1].reshape(-1,1)),axis=1)
        target = Data[:, -1].reshape(-1,1)
        
        target, qt = anamor(np.concatenate( (Grid_Data, target) ,axis=1) ) 
       
        # Interpolate Geochemical Data to the grid dimension
        Radius=(Ymax-Ymin)/Lat/2
        
        idw_tree = idw.tree(Grid_Data, target)            
        v = idw_tree(Grid, k=min(10,np.size(Data[:,-1])), width=Radius, pp=2).reshape(-1,1)
        loc=v>-1000
        EV_Data=EV[loc.ravel(),:]
        target=v[loc]
        print(np.shape(target)) 
        
        p_train, p_test, target_train, target_test = train_test_split(
        EV_Data, target, test_size=0.01, random_state=8958)

        m_rk = models
        m_rk.fit(p_train, target_train.ravel())
        Map=m_rk.predict(EV).reshape(-1,1)
        
        Map=anamorinv(qt,Map)
        
        interpolationMap=np.concatenate((interpolationMap,Map),axis=1)
        

    Path ="ProcessData\\SedimentGeochemistry" + "\\" + Feuillet + ".csv"
    df = pd.DataFrame(interpolationMap, columns = title)  
    df.to_csv(Path, index=False)     
        
    return df         
# %%
def RegressionKrigingGeoRock(Feuillet, Element):


    models = RandomForestRegressor(n_estimators=400, random_state=85465, bootstrap=True, max_depth=None,
                                   min_samples_split=2, min_samples_leaf=1, max_features = 'auto')
        
    title=["Coord_X","Coord_Y","R_IAI","R_CCPI","R_VMSI","R_AAAI",
           "R_K2O","R_MGO","R_NA2O","R_CAO","R_FE2O3","R_MNO","R_SIO2",
           "R_TIO2","R_AL2O3","R_P2O5",
           "R_AS","R_AU","R_AG","R_CU","R_SB","R_W","R_MO","R_NI","R_ZN","R_PB"]

    Element_ID= [ "Coord_X", "Coord_Y",
	        "MagHR", "MagGradHR", "SigAnoMagHR", "BouguerBR", "AnoGrav1DVBR",
            "Contact_Binairy", "Contact_Density", "Contact_Distance",
            "Fault_Binairy", "Fault_Density",  "Fault_Distance", 
	        "GeoRegio"]

    print(Feuillet)
    # InputData for regression Kriging
    Path ="ProcessData\\GeoStruct\\" + Feuillet + ".csv"
    df = pd.read_csv(Path)
    InputData =  df[Element_ID].to_numpy()
    EV = InputData[:, 2:]
    
    # Define Domain Dimension 
    Xmin, Xmax, Ymin, Ymax, Lat, Long=Dimension(Feuillet)
    
    # Grid dimension
    xv, yv = np.meshgrid(np.linspace(Xmin, Xmax, Long),np.linspace(Ymax, Ymin, Lat) ,)
    Grid=np.concatenate((xv.reshape(-1,1), yv.reshape(-1,1)), axis=1)   
    
    # Tranform data to the original space
    interpolationMap=InputData[:,0:2]

    # Loading geochimical Data   
    Path ="RawData\\RockGeochemistry" + "\\" + Feuillet + ".csv"
    TableData = pd.read_csv(Path)

    # interpolation of each component
    for chim in Element:
        print(chim)
        # Selection of the element
        Data=TableData[["Coord_X","Coord_Y",chim]]
        Data=pd.DataFrame(Data).to_numpy() 
        if chim in ["K2O","MGO","NA2O","CAO","FE2O3","MNO","SIO2","TIO2","AL2O3","P2O5"]:
            Data=Data[Data[:,-1]>0 ,:]
            code=2
        else:
            Data=Data[Data[:,-1]>=0 ,:]
            code=-2
            
        Grid_Data=np.concatenate((Data[:, 0].reshape(-1,1),Data[:, 1].reshape(-1,1)),axis=1)
        target = Data[:, -1].reshape(-1,1)
                        
       
        # Interpolate Geochemical Data to the grid dimension
        Radius=(Ymax-Ymin)/Lat/2
        
        idw_tree = idw.tree(Grid_Data, target)            
        v = idw_tree(Grid, k=min(10,np.size(Data[:,-1])), width=Radius, pp=code).reshape(-1,1)
        loc=v>-1000
        EV_Data=EV[loc.ravel(),:]
        target=v[loc]
        print(np.shape(target)) 
        
        
        target, qt = anamor(np.concatenate( (Grid[loc.ravel(),:], target.reshape(-1,1)) ,axis=1) ) 
                    
        p_train, p_test, target_train, target_test = train_test_split(
                                            EV_Data, target, test_size=0.01, random_state=8958)
        
        #best_params, best_scores, models = Classifier_PMP_Tuning(p_train, target_train, seed=15421, categ=0)  
        
        m_rk= models
        #m_rk = RandomForestRegressor(n_estimators=best_params['n_estimators'],
        #                            min_samples_split=best_params['min_samples_split'],
        #                            min_samples_leaf=best_params['min_samples_leaf'],
        #                            max_features=best_params['max_features'], max_depth=best_params['max_depth'],
        #                            criterion=best_params['criterion'], bootstrap=best_params['bootstrap'],
        #                            n_jobs=-1, random_state=85465)
        
        m_rk.fit(p_train, target_train.ravel())
        score=m_rk.score(EV_Data, target)
        print(score)
        score=m_rk.score(p_train, target_train)
        print(score)
        score=m_rk.score(p_test, target_test)
        print(score)
        Map=m_rk.predict(EV).reshape(-1,1)               

        Map=anamorinv(qt,Map)
        
        interpolationMap=np.concatenate((interpolationMap,Map),axis=1)



    xv, yv = np.meshgrid(np.linspace(0, Long-1, Long), np.linspace(Lat-1, 0, Lat))
    ImageV=np.concatenate((xv.reshape(-1,1), yv.reshape(-1,1)), axis=1)
    
    
    interpolationMapAll= ((np.sum(interpolationMap[:,[2,3]],axis=1)) / (np.sum(interpolationMap[:,[2,3,4,5]],axis=1)) ).reshape(-1,1)
    interpolationMapAll=np.concatenate((interpolationMapAll, ((np.sum(interpolationMap[:,[6,3]],axis=1)) / (np.sum(interpolationMap[:,[6,3,4,2]],axis=1))).reshape(-1,1) ), axis=1 )
    interpolationMapAll=np.concatenate((interpolationMapAll, ((np.sum(interpolationMap[:,[3,6,7]],axis=1)) / (np.sum(interpolationMap[:,[2,4,5]],axis=1))).reshape(-1,1) ), axis=1 )
     
    interpolationMapAll=np.concatenate( (interpolationMapAll, ((interpolationMap[:,8]) / ( interpolationMap[:,8]+
                                                            interpolationMap[:,3]*10+interpolationMap[:,4]*10+interpolationMap[:,5]*10 ) ).reshape(-1,1)  ) , axis=1 )
    interpolationMapAll=np.concatenate((interpolationMapAll,interpolationMap[:,2:] ), axis=1  )

        
    interpolation=np.hstack((ImageV, interpolationMapAll))
    
    Path ="ProcessData\\RockGeochemistry" + "\\" + Feuillet + ".csv"
    df = pd.DataFrame(interpolation, columns = title)  
    df.to_csv(Path, index=False) 

    return df 
