# -*- coding: utf-8 -*-
'''
Created on Oct 6, 2015

@author: sebastian
'''

from numpy import arange, asarray, exp
from scipy.optimize.minpack import curve_fit
from util.load_static_data import loadLookupTables

lookUpTablePath = "../data/lookup_tables/"              # Path to LookUpTable-Folder
luts            = loadLookupTables(lookUpTablePath)     # Load Lookup-Tables

# Script for fitting of a function to some data distribution (used for Lookup-Table approximation)
# Theoretical model (potential equation; 2 independent variables x and y)
def func(X, a, b, c):
    x, y = X
    return exp(a) * pow(x, b) * pow(y, c)

# Method for fitting "lut_cp_cirrus" to quadratic function:
def fitCirrusLutToQuadFunc(lut_cp_cirrus):
    # data to fit
    x = arange(1.,2.1,0.25)
    y = arange(220.,291.,5.)
    
    xs = []
    for i in range(len(y)): xs.extend(x)
    ys = []
    for j in range(len(y)): 
        for i in range(len(x)): ys.append(y[j])
    zs = []
    for row in lut_cp_cirrus:
        for i in range(len(row)-1): zs.append(row[i+1])
            
    xs = asarray(xs)
    ys = asarray(ys)
    zs = asarray(zs)
    
    # initial guesses for a,b,c:
    p0 = 1., 1., 1.
    return curve_fit(func, (xs,ys), zs, p0)

def result(a,b,c):
    print func((1.00,220.),a,b,c)
    print func((1.25,220.),a,b,c)
    print func((1.50,220.),a,b,c)
    print func((1.75,220.),a,b,c)
    print func((2.00,220.),a,b,c)
    print "\n"
    print func((1.00,260.),a,b,c)
    print func((1.25,260.),a,b,c)
    print func((1.50,260.),a,b,c)
    print func((1.75,260.),a,b,c)
    print func((2.00,260.),a,b,c)
    print "\n"
    print func((1.00,290.),a,b,c)
    print func((1.25,290.),a,b,c)
    print func((1.50,290.),a,b,c)
    print func((1.75,290.),a,b,c)
    print func((2.00,290.),a,b,c)
    return

fit = fitCirrusLutToQuadFunc(luts[0])
print fit[0][0],fit[0][1],fit[0][2]
result(fit[0][0],fit[0][1],fit[0][2])
