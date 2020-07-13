#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 23:48:32 2020

@author: lamho
"""

from scipy.integrate import odeint
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def SEIR(y, t, logbeta, logkappa, loggamma):
    # S: y[0]
    # E: y[1]
    # I: y[2]
    # R: y[3]
    beta, kappa, gamma = np.exp((logbeta, logkappa, loggamma))
    
    return np.array([- beta*y[0]*y[2] / N, 
                     beta*y[0]*y[2] / N - kappa*y[1], 
                     kappa*y[1] - gamma*y[2], 
                     gamma*y[2]])
    
def minimization(y0, t, I, niter = 1):
    
    def fit_odeint(t, logbeta, logkappa, loggamma):
        return odeint(SEIR, y0, t, args=(logbeta, logkappa, loggamma))[:,2]
    
    best = np.inf
    res = (0, 0, 0)
    for i in range(niter):
        init_logbeta = 0.5 * np.random.randn() + 1
        init_logkappa = 0.5 * np.random.randn()
        init_loggamma = 0.5 * np.random.randn()
        try:
            popt, pcov = curve_fit(fit_odeint, t, I,
                                   p0=np.asarray([init_logbeta,init_logkappa,init_loggamma]),
                                   maxfev=5000)
        except RuntimeError:
            print("Error - curve_fit failed")   
        fitted = fit_odeint(t, *popt)
        value = np.sum((fitted - I)**2)
        if (value < best): 
            res = popt
            best = value
    return (res, best)

def dynamics(y0, t, logbeta, logkappa, loggamma):
    return odeint(SEIR, y0, t, args=(logbeta, logkappa, loggamma))