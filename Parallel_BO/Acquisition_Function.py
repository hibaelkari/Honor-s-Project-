# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:34:12 2023

@author: Hiba
"""

import scipy
import numpy as np

# FIND FIRST POINT BY MAXIMIZING EI_MAX OR EI_MIN

def EI_MAX(muNew, stdNew, fMax,epsilon =0.01):

    
    Z = (muNew - fMax )/stdNew
    if (stdNew == 0):
     val = 0.0 
    else:
     val = (muNew - fMax )* scipy.stats.norm.cdf(Z) + stdNew*scipy.stats.norm.pdf(Z)

    return val

    



	

        
#####################################################################################################        

def EI_MIN(muNew, stdNew, fMin,epsilon =0.01):

    
    Z = (fMin - muNew )/stdNew
    if (stdNew == 0):
     val = 0.0 
    else:
     val = (fMin - muNew )* scipy.stats.norm.cdf(Z) + stdNew*scipy.stats.norm.pdf(Z)

    return val

########################################################################################################################



##################################################################################################
# INFLUENCE FUNCTION DEFINITION at the qth point : I THINK THE FUNCTION CONTAINS A LOT OF MISTAKES!!
def IF ( previous_points_vector,muNew, sigma): #not sure if we should include also muNew
    sum = 0 
    num_of_previous_points = len(previous_points_vector)
    for i in range (0,num_of_previous_points):
        sum = sum + np.exp(-0.5*(muNew - previous_points_vector[i])* np.inv(sigma)  *np.transpose(muNew - previous_points_vector[i]))



#############################################################################################

# pseudo expected improvement for minimization

def PEI_MIN (previous_points_vector,muNew,sigma, stdNew, fMin):
    influence = IF(previous_points_vector , muNew, sigma)
    EI = EI_MIN(muNew, stdNew, fMin,epsilon = 0.01)
    
    return influence*EI

#####################################################################################

# pseudo expected improvement for maximization

def PEI_MAX (previous_points_vector,muNew,sigma, stdNew, fMax):
    influence = IF(previous_points_vector , muNew, sigma)
    EI = EI_MAX(muNew,stdNew,fMax,epsilon = 0.01)
             
    return influence*EI

