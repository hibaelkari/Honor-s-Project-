# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:26:59 2023

@author: Hiba
"""
import pyDOE2
import numpy as np
import scipy
import math 


# ## example to apply on vectors 

# vec = [1,2,3]
# sigma = np.diag(vec)
# x1 = np.array([1,2,3])
# x2= np.array([1,1,3])

# x1 = x1.reshape(-1,1)
# x2 = x2.reshape(-1,1)

# x1 = np.transpose(x1)
# x2 = np.transpose(x2)

# function that takes 2 x values and computes the correlation between them based on RBF equation 
def Correlation_Func (x1, x2, sigma):  #x1 and x2 sould be inputted as row vectors of size [1, nb of design variables ]
    
        f1 = x1 - x2   # row vector 
        f2 = np.linalg.inv(sigma)
        f3 = np.transpose(x1-x2)
        fac1 = np.linalg.multi_dot([f1,f2,f3])
        
        corr = math.exp(-0.5*fac1)
    
        return corr





## example to apply on scalars


# ress = Correlation_Function(x1,x2,sigma)
# x1_new = np.array([1])
# x2_new = np.array([9])
# x1_new = x1_new.reshape(-1,1)
# x2_new = x2_new.reshape(-1,1)
# sigma = np.array([1])
# sigma = sigma.reshape(-1,1)
# ress_scalar = Correlation_Function(x1_new,x2_new,sigma)


