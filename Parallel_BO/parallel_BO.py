# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:20:07 2023

@author: Hiba
"""

# This code is an implementation of the parallel BO (triple enginer global BO)
import scipy
import lhsmdu
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from Acquisition_Function.py import optimize_EI_MAX, optimize_EI_MIN
# some initializations:
nsamples = 4  # the number of samples for initializing the LHS DOE

# step one: Implement a Latin Hypercube Sampling Scheme
    # Originally, we have 4 input parameters (4d optimization problem)
l = lhsmdu.sample(4,4) # Latin Hypercube Sampling of four variables, and 4 samples each

# formulate the X matrix:
X = np.transpose(l)

# step two: For each alpha combination (ie: each column), we have to run the simulation and get Fmean
    # Here, you should call the objective function
# Y = np.zeros(nsamples)
# for i in range (0,nsamples):
#     Ai = l[:,i]
#     Fmean = Objective_Func(Ai)
#     # formulate the Y matrix:
#     Y[i] = Fmean
Y = [100,132,134,129]
# CREATE A FUNCTION THAT CREATES THE GPR MODEL AND OUTPUTS FROM IT THE GPR ITSELF
gpr = GaussianProcessRegressor(kernel = RBF(length_scale = np.ones(4), length_scale_bounds=((1e-5,1e5),(1e-5,1e5),(1e-5,1e5),(1e-5,1e5))),alpha=1e-10,optimizer='fmin_l_bfgs_b',n_restarts_optimizer=5, normalize_y=False, copy_X_train=True, random_state=None).fit(X,Y)
# output the hyper parameters from the gpr model results because we need them in computing the 
#pseudo expected improvement, this is only applied for the length scale values in thiscase :)

# AFTER CALLING THE FUNCTION; WE CAN EXTRACT THESE VALUES FROM ITS OUTPUT
hyper_params =  gpr.L_[3,:]

# define sigma, which is needed to compute the P-EI

sigma = np.diag(hyper_params)

# to find approximates y value given an x value
yhat = gpr.predict(X)

#############################
# step two: Find the max of EI_min and EI_max to see which engine to choose

#Now, we want to find the maximum of EI_min and EI_max

res1 = scipy.optimize.minimize(optimize_EI_MIN, x0 = np.ones(3), args=(), method='L-BFGS-B', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)

# input optimized results
x_value_1 = res1.x

#function value at the optimum
f_value_1 = res1.fun

delta_y_1 = f_value_1
#####################################################################################

res2 = scipy.optimize.minimize(optimize_EI_MAX, x0 = np.ones(3), args=(), method='L-BFGS-B', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)

# input optimized results
x_value_2 = res2.x

#function value at the optimum
f_value_2 = res2.fun

delta_y_2 = f_value_2

############################

## evaluate the values that decide which engine to choose:
counter = 0
list = np.linspace(0,1,10)
y = np.zeros(10)
for x in list:
    # evaluate the GPR model
    y[counter] = gpr.predict(x)
    counter = counter + 1
delta = 1e-6
current_y_min = np.argmin(y)
current_y_max = np.argmax(y)

v1 = delta_y_1 / (np.abs(current_y_min) + delta)
v2 = delta_y_2 / (np.abs(current_y_max) + delta)


######### define the different cases and which engine to choose
