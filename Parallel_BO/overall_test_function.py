# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:32:40 2023

@author: Hiba
"""
import scipy
import numpy as np
import math
import lhsmdu
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Hyperparameter
import matplotlib.pyplot as plt
from Acquisition_Function import EI_MAX, EI_MIN
import pyDOE2
from Correlation_Function import Correlation_Func 


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


y_min_current = np.min(mean_prediction)
y_max_current = np.max(mean_prediction)

# # find the expected improvement values:
EI_max_values = np.zeros(len(mean_prediction))
EI_min_values = np.zeros(len(std_prediction))

for i in range (0, len(mean_prediction)):
                
   EI_max_values[i] =  EI_MAX(mean_prediction[i], std_prediction[i], y_max_current,epsilon =0.01)
   EI_min_values[i] =  EI_MIN (mean_prediction[i], std_prediction[i], y_min_current, epsilon = 0.01)
    
   
# now: find the  max of EImin and max of EImax
max_EI_max = np.max(EI_max_values)
max_EI_min = np.max(EI_min_values)

deltay1 = max_EI_min
deltay2 = max_EI_max

delta = 1e-6 

v1 = deltay1 / (np.abs(y_min_current) + delta)
v2 = deltay2 / (np.abs(y_max_current) + delta)
v1 = 0.5
v2 = 0.00001
# setting up the 6 different critria:

# Criteria 1:
if (v1 < eps_min and v2 < eps_max) : 
    print('No need for more points to find both:  the minimum and maximum')
    counter = counter + 1
    if (counter == 2):
      print ('The optima have been found')
      # terminate the program here , or can put all this in a while loop (the part of the engines)
      # else, continue checking the criteria
    else: # here counter not equal to 2, probably = 1
      print('Possibility of fake convergence')
      # Use Engine 3
#Criteria 2:
if (v1 >= eps_min and v2>= eps_max):
    print('GPR is still not accurate for both: the min and the max')
    # Use Engine 3

# Criteria 3: Incorprated with criteria 1

# Criteria 4: 
    
if (v1>= eps_min and v2 < eps_max):
    print('GPR still not accurate to estimate the minimum')
    # Use Engine 1
    
    # first step
    xnew = np.zeros(q)
    min_step_0 = np.max(EI_min_values)
    index_at_step_0= list(EI_min_values).index(min_step_0)
    x_at_step_0 = xval[int(index_at_step_0)]
    xnew[0] = x_at_step_0
    # other steps
    
    #### STEP ONE
    x_for_improvement = xnew[0]
    x_for_improvement = x_for_improvement.reshape(-1,1)
        
    ## FOR LOOP START
    
    # find the influence function for all x's in x_for_improvement
        # initialize IF to 1
    for k in range (1,q):
        influence_vector = np.zeros(1000)
        PEI_min_for_each_xval = np.zeros(1000)
        for i in range (0,len(xval)):
            IF = 1
        
            for j in range (0, len(x_for_improvement)):
        
        #x2 = np.array(xval[0])
                x1 = x_for_improvement[j].reshape(-1,1)
                x2 = xval[i].reshape(-1,1)
                sigma_matrix = sigma_matrix.reshape(-1,1)
                IF = IF * (1- Correlation_Func(x1, x2, sigma_matrix) )
            # create IF vector for all the control points:
            influence_vector[i] = IF
            PEI_min_for_each_xval [i] = IF * EI_min_values[i]
# Now we can maximize PEI_min 
        min_step_1 = np.max(PEI_min_for_each_xval)
        index_at_step_1= list(PEI_min_for_each_xval).index(min_step_1)
        x_at_step_1 = xval[int(index_at_step_1)]

        xnew[k] = x_at_step_1

## Now we should repeat the whole procedure, but now , using a new x_for_improvement
        x_for_improvement = xnew[0:k+1]

# Now, given x_for_improvement values, we want to evaluate the objective function at these
# points and then fit a new gpr model
y_for_improvement = np.zeros(q)
for h in range (0, q):
    y_for_improvement[h] = Objective_Function(x_for_improvement[h])
x_for_improvement = x_for_improvement.reshape(-1,1)
y_for_improvement = y_for_improvement.reshape(-1,1)
# now create the new gpr and fit in again:
Xnew = np.concatenate([X,x_for_improvement])
Ynew = np.concatenate([Y, y_for_improvement])
#gpr_new = GaussianProcessRegressor(kernel =1*RBF(length_scale = np.ones(1), length_scale_bounds=(1e-5,1e5)),alpha=1e-10,optimizer='fmin_l_bfgs_b',n_restarts_optimizer=10, normalize_y=False, copy_X_train=True, random_state=None).fit(Xnew,Ynew)


# # try to plot the kernel:
# xval  = np.linspace(start=0, stop=1, num=1_000).reshape(-1, 1)
# mean_prediction, std_prediction = gpr_new.predict(xval, return_std=True)
# std_prediction = std_prediction.reshape(-1,1)
# mean_prediction = mean_prediction.reshape(-1,1)
# xval = np.array(xval)
# plt.plot(xval, mean_prediction, label="Mean prediction")
# plt.fill_between(
#        xval.ravel() , 
#        list((mean_prediction - 1.96 * std_prediction).flatten()),
#        list((mean_prediction + 1.96 * std_prediction).flatten()),
#        alpha=0.5,
#        label=r"95% confidence interval",
#   )
# plt.legend()
# plt.xlabel("$x$")
# plt.ylabel("$f(x)$")
# _ = plt.title("Gaussian process regression model")    


        
      

 
# Criteria 5:
if (v1 < eps_min and v2 >= eps_max):
    print('GPR still not accurate to estimate the maximum')
    # Use Engine 2

