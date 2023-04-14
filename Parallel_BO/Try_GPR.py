# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:13:11 2023

@author: Hiba
"""


import numpy as np
import math
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Hyperparameter
import matplotlib.pyplot as plt
from Acquisition_Function import EI_MAX, EI_MIN
import pyDOE2



# First define the objective function:

def Objective_Function (x):  # x is between 0 and 1
        
   y  = (2*x - 1)**2 * math.sin( (4*math.pi*x ) -  (math.pi / 8)  )
    
   return y

# Parallel_BO input parameters:

# feasible region:
xI = [0,1] # default in LHS, but should modify LHS to be between [-1,1] in later stages
# initial sample size:
nO =10
# epsilon min
eps_min = 0.002 
#epsilon max
eps_max = 0.002

# to check if we have converged for 2 successive iterations when using the engines    
counter = 0    
   
# q for the engines
q = 8 
# Next: Perform a LHS, generate an initial training data set



num_dv = 1               # dimension, here it is a 1D problem 
num_samp =  10        # No. of samples, nO : here chosen = 5
#num_iterations=5000     # number of iterations for improving LH (not sure if we need it)

num_sampf = float(num_samp) # turn integer to float to avoid getting zero in devision  

## -------------------------- Original LHS
## by default, lower bound: 0, upper bound: 1  (ask how to change them ? by reshape??)
DoE_LHS = pyDOE2.lhs(num_dv, num_samp,criterion = 'centermaximin') 
X = np.zeros(len(DoE_LHS))
for i in range (0, len(DoE_LHS)):
    arr = DoE_LHS[i]
    X[i] = arr[0]





Y = np.zeros(num_samp)
#deduce the y matrix:
for i in range (0,nO):
    
    # formulate the Y matrix:
    Y[i] = Objective_Function(X[i])


X = X.reshape(-1,1)
Y = Y.reshape(-1,1)   
# Fit a GPR model using the training data:
gpr = GaussianProcessRegressor(kernel =1*RBF(length_scale = np.ones(1), length_scale_bounds=(1e-5,1e15)),alpha=1e-10,optimizer='fmin_l_bfgs_b',n_restarts_optimizer=5, normalize_y=False, copy_X_train=True, random_state=None).fit(X,Y)
new_kernel = gpr.kernel_

# extract the length scale vector from the new kernel
kernel_params_dictionary = new_kernel.get_params(deep= True)
li_vector = kernel_params_dictionary["k2__length_scale"]

# find sigma: the diagonal matrix that includes th elength scale values on its digaonal
li_vector = li_vector.reshape(-1,1)
sigma_matrix = np.diag(li_vector)

# try to plot the kernel:
xval  = np.linspace(start=0, stop=1, num=1_000).reshape(-1, 1)
mean_prediction, std_prediction = gpr.predict(xval, return_std=True)
std_prediction = std_prediction.reshape(-1,1)
mean_prediction = mean_prediction.reshape(-1,1)
xval = np.array(xval)
plt.plot(xval, mean_prediction, label="Mean prediction")
plt.fill_between(
      xval.ravel() , 
      list((mean_prediction - 1.96 * std_prediction).flatten()),
      list((mean_prediction + 1.96 * std_prediction).flatten()),
      alpha=0.5,
      label=r"95% confidence interval",
  )
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Gaussian process regression model")