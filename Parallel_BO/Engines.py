# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:55:45 2023

@author: Hiba
"""
from Acquisition_Function.py import PEI_MIN, PEI_MAX, EI_MIN, EI_MAX
import numpy as np
# for now: set q = 8 points
q = 8 
# make sure of the input arguments of all the engines
# Engine 1:

def Engine_1 (q, previous_points_vector, muNew, sigma, stdNew,fMin):
    xnew = np.zeros(q)
    xnew[0] = np.argmax(EI_MIN)   # choose a maximization algorithm for that
    for i in range(1,q):
        xnew[i] = np.argmax(PEI_MIN(previous_points_vector,muNew,sigma, stdNew, fMin))  #choose a maximization algorithm for that
        
        return xnew

def Engine_2(q, previous_points_vector, muNew, sigma, stdNew,fMax):
    xnew = np.zeros(q)
    xnew[0] = np.argmax(EI_MAX) # choose a maximization algorithm for that
    for i in range (1,q):
        xnew[i] = np.argmax(PEI_MAX(previous_points_vector,muNew,sigma, stdNew, fMax))
        return xnew
    

def Engine_3(q,previous_points_vector,muNew,sigma, stdNew, fMax, fMin):
    xnew = np.zeros(q)
    xnew[0] = np.argmax(EI_MIN)   # choose a maximization algorithm for that
    xnew[1]=  np.argmax(EI_MAX) # choose a maximization algorithm for that
    
    for i in range (2,q):
        if (i%2 == 0): # ie : this is an even number 
            xnew[i]= np.argmax(PEI_MIN(previous_points_vector,muNew,sigma, stdNew, fMin))  #choose a maximization algorithm for that
        
        else: # ie this is an odd number 
            xnew[i]= np.argmax(PEI_MAX(previous_points_vector,muNew,sigma, stdNew, fMax))
            
    return xnew
        