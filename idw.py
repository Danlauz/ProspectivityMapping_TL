# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:25:46 2023

@author: dany-
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2016 Paul Brodersen <paulbrodersen+idw@gmail.com>

# Author: Paul Brodersen <paulbrodersen+idw@gmail.com>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
Inverse distance weighting (IDW)
--------------------------------
Compute the score of query points based on the scores of their k-nearest neighbours,
weighted by the inverse of their distances.
Example:
--------
# import idw
# 'train'
idw_tree = idw.tree(X1, z1)
# 'test'
spacing = np.linspace(-5., 5., 100)
X2 = np.meshgrid(spacing, spacing)
grid_shape = X2[0].shape
X2 = np.reshape(X2, (2, -1)).T
z2 = idw_tree(X2)
For a more complete example see demo().
"""

import numpy as np
from scipy.spatial import cKDTree

class tree(object):
    """
    Compute the score of query points based on the scores of their k-nearest neighbours,
    weighted by the inverse of their distances.
    @reference:
    https://en.wikipedia.org/wiki/Inverse_distance_weighting
    Arguments:
    ----------
        X: (N, d) ndarray
            Coordinates of N sample points in a d-dimensional space.
        z: (N,) ndarray
            Corresponding scores.
        leafsize: int (default 10)
            Leafsize of KD-tree data structure;
            should be less than 20.
    Returns:
    --------
        tree instance: object
    Example:
    --------
    # 'train'
    idw_tree = tree(X1, z1)
    # 'test'
    spacing = np.linspace(-5., 5., 100)
    X2 = np.meshgrid(spacing, spacing)
    X2 = np.reshape(X2, (2, -1)).T
    z2 = idw_tree(X2)
    See also:
    ---------
    demo()
    """
    def __init__(self, X=None, z=None, leafsize=10):
        if not X is None:
            self.tree = cKDTree(X, leafsize=leafsize )
        if not z is None:
            self.z = np.array(z)

    def fit(self, X=None, z=None, leafsize=10):
        """
        Instantiate KDtree for fast query of k-nearest neighbour distances.
        Arguments:
        ----------
            X: (N, d) ndarray
                Coordinates of N sample points in a d-dimensional space.
            z: (N,) ndarray
                Corresponding scores.
            leafsize: int (default 10)
                Leafsize of KD-tree data structure;
                should be less than 20.
        Returns:
        --------
            idw_tree instance: object
        Notes:
        -------
        Wrapper around __init__().
        """
        return self.__init__(X, z, leafsize)

    def __call__(self, X, k=10, eps=1e-8, p=2, regularize_by=1e-9, width=0.1, pp=2):
        """
        Compute the score of query points based on the scores of their k-nearest neighbours,
        weighted by the inverse of their distances.
        Arguments:
        ----------
            X: (N, d) ndarray
                Coordinates of N query points in a d-dimensional space.
            k: int (default 6)
                Number of nearest neighbours to use.
            p: int or inf
                Which Minkowski p-norm to use.
                -1 : Neirest data
                0 : Average
                1 : sum-of-absolute-values "Manhattan" distance
                2 : Euclidean distance
                infinity is the maximum-coordinate-difference distance
            eps: float (default 1e-6)
                Return approximate nearest neighbors; the k-th returned value
                is guaranteed to be no further than (1+eps) times the
                distance to the real k-th nearest neighbor.
            regularise_by: float (default 1e-9)
                Regularise distances to prevent division by zero
                for sample points with the same location as query points.
        Returns:
        --------
            z: (N,) ndarray
                Corresponding scores.
        """
        self.distances, self.idx = self.tree.query(X, k, eps=eps, p=p)
        self.distances += regularize_by
        
        weights = self.z[self.idx.ravel()].reshape(self.idx.shape)
        


        if pp==0: #Mean
            IDX= (self.distances <= width)
            IDXNot= (self.distances > width)
            
            weights[IDXNot]=0
            self.distances[IDX]=1
            self.distances[IDXNot]=0
                                    
            sum_dist=np.sum(self.distances, axis=1)
            sum_dist[sum_dist==0]=1
            mw = np.sum(weights, axis=1) / sum_dist 
            mw[np.sum(self.distances, axis=1)==0]=-1000
            
        elif pp==-1 : #Nearest
            weights[self.distances > width]=0
            mw = weights[:,0]
            mw[np.sum(self.distances, axis=1)==0]=-1000
            
        elif pp==-2 : #Maximum
            IDX= (self.distances <= width)
            IDXNot= (self.distances > width)
            
            weights[IDXNot]=0
            self.distances[IDX]=1
            self.distances[IDXNot]=0
        
            mw = np.max(weights,axis=1) 
            mw[np.sum(self.distances, axis=1)==0]=-1000
            
        else:
            weights[self.distances > width]=0
            self.distances[self.distances > width]=1E6                         
            sum_dist=np.sum(1./self.distances**pp, axis=1)
            mw = np.sum(weights/self.distances**pp, axis=1) / sum_dist
            
            mw[np.sum(self.distances<width, axis=1)==0]=-1000
                       
        return mw

    def transform(self, X, k=10, p=2, eps=1e-8, regularize_by=1e-9, width=0.1, pp=2):
        """
        Compute the score of query points based on the scores of their k-nearest neighbours,
        weighted by the inverse of their distances.
        Arguments:
        ----------
            X: (N, d) ndarray
                Coordinates of N query points in a d-dimensional space.
            k: int (default 6)
                Number of nearest neighbours to use.
            p: int or inf
                Which Minkowski p-norm to use.
                1 is the sum-of-absolute-values "Manhattan" distance
                2 is the usual Euclidean distance
                infinity is the maximum-coordinate-difference distance
            eps: float (default 1e-6)
                Return approximate nearest neighbors; the k-th returned value
                is guaranteed to be no further than (1+eps) times the
                distance to the real k-th nearest neighbor.
            regularise_by: float (default 1e-9)
                Regularise distances to prevent division by zero
                for sample points with the same location as query points.
        Returns:
        --------
            z: (N,) ndarray
                Corresponding scores.
        Notes:
        ------
        Wrapper around __call__().
        """
        return self.__call__(X, k, eps, p, regularize_by)
