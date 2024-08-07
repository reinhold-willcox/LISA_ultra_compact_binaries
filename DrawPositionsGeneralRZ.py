#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:05:24 2024

@author: alexey
"""

import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import root
import sys
sys.path.insert(1, './PyModules/')

#Units are MSun, kpc, Gyr

MGal = 6.43e10 #From Licquia and Newman 2015


BesanconParams = {10:{'Eps':0.0791}}

#Rho(r,z), non-normalized for the Besancon model, thin disk
def RhoBesanconPre(r,z,iBin):
    hPlus  = 2.53
    hMinus = 1.32
    Eps    = BesanconParams[iBin]['Eps']
    aParam = np.sqrt(r**2 + z**2/Eps**2)
    Res    = np.exp(-(0.5**2 + aParam**2/hPlus**2)**0.5) - np.exp(-(0.5**2 + aParam**2/hMinus**2)**0.5)
    return Res

GalFunctionsDict = {'Besancon': RhoBesanconPre}

def GetRhoBar(r,iBin,Model):
    Nz      = 300
    ZSet    = np.linspace(0,2,Nz)
    RhoFun  = GalFunctionsDict[Model]
    RhoSet  = np.zeros(Nz)
    for i in range(Nz):
        RhoSet[i] = RhoFun(r,ZSet[i],iBin)
    RhoBar  = np.sum(RhoSet)
    return RhoBar

def GetZ(r,iBin,Model):
    Nz      = 300
    ZSet    = np.linspace(0,2,Nz)
    RhoFun  = GalFunctionsDict[Model]
    RhoSet  = np.zeros(Nz)
    for i in range(Nz):
        RhoSet[i] = RhoFun(r,ZSet[i],iBin)
        
    MidZSet   = 0.5*(ZSet[1:] + ZSet[:-1])
    DeltaZSet = 0.5*(ZSet[1:] - ZSet[:-1])
    MidRhoSet = 0.5*(RhoSet[1:] + RhoSet[:-1])
    RhoBar    = np.sum(MidRhoSet*DeltaZSet)
    RhozCDF   = np.cumsum(MidRhoSet*DeltaZSet)/RhoBar
    
    Xiz        = np.random.rand()
    SignXi     = np.sign(2*(np.random.rand() - 0.5))
    zFin       = SignXi*np.interp(Xiz,RhozCDF,MidZSet)    
    return zFin
    
           

def RhoRArray(iBin, Model):
    Nr     = 1000
    RSet   = np.linspace(0,30,Nr)
    RhoSet = np.zeros(Nr)
    for i in range(Nr):
        RCurr      = RSet[i]
        RhoSet[i]  = GetRhoBar(RCurr,iBin,Model)
        
    Res = {'RSetKpc':RSet, 'RhoSet': RhoSet}
    return Res

def PreCompute(iBin, Model):
    RhoRDict   = RhoRArray(iBin, Model)
    RSet       = RhoRDict['RSetKpc']
    RhoRSet    = RhoRDict['RhoSet']
    MidRSet    = 0.5*(RSet[1:] + RSet[:-1])
    DeltaRSet  = RSet[1:] - RSet[:-1]
    MidRhoSet  = 0.5*(RhoRSet[1:] + RhoRSet[:-1])
    RhoNorm    = 2*np.pi*np.sum(MidRSet*DeltaRSet*MidRhoSet)
    RCDFSet    = 2*np.pi*np.cumsum(MidRSet*DeltaRSet*MidRhoSet)/RhoNorm
    
    Res        = {'MidRSet': MidRSet, 'RCDFSet': RCDFSet}
    return Res

ModelCache     = PreCompute(10,'Besancon')
    
def DrawRZ(iBin,Model):
    MidRSet    = ModelCache['MidRSet']
    RCDFSet    = ModelCache['RCDFSet']
    
    Xir        = np.random.rand()
    RFin       = np.interp(Xir,RCDFSet,MidRSet)
    zFin       = GetZ(RFin,iBin,Model)
    
    
    return [RFin,zFin]



    def RCDFInv(Xir,Hr):    
        # Get the parameters for the inverse CDF
        def RCD(R):
            Res = (1-np.exp(-R/Hr))-(R/Hr)*np.exp(-R/Hr)-Xir
            return Res
        
        Sol  = sp.optimize.root_scalar(RCD,bracket=(0.0001*Hr,20*Hr))
        if Sol.converged:
            R      = Sol.root
        else:
            print('The radial solution did not converge')
            sys.exit()
        return R
    
NPoints = 10000

RSetFin = np.zeros(NPoints)
ZSetFin = np.zeros(NPoints)

for i in range(NPoints):
    ResCurr     = DrawRZ(10,'Besancon')
    RSetFin[i]  = ResCurr[0]
    ZSetFin[i]  = ResCurr[1]
    

Th       = np.asarray([2.*np.pi*np.random.uniform() for i in range(NPoints)],dtype=float)
X        = RSetFin*np.cos(Th)
Y        = RSetFin*np.sin(Th)
Age      = np.random.uniform(0, 12)
 
XRel     = X - 8.
YRel     = Y
ZRel     = ZSetFin

RRel     = np.sqrt(XRel**2 + YRel**2 + ZRel**2)

ResDict  = {'Age': Age, 'Xkpc': X, 'Ykpc': Y, 'Zkpc': ZSetFin, 'Rkpc': RSetFin, 'Th': Th, 'XRelkpc': XRel, 'YRelkpc':YRel, 'ZRelkpc': ZRel, 'RRelkpc': RRel}
ResDF    = pd.DataFrame(ResDict)    
ResDF.to_csv('./GalTest2.csv', index = False)

sys.exit()




def Sample1D(Hr,Hz):
    RRand    = np.random.uniform()
    ZRand    = np.random.uniform()
    ZSign    = np.sign(np.random.uniform() - 0.5)
    R        = RCDFInv(RRand,Hr)
    Z        = zCDFInv(ZRand,Hz)*ZSign
    Th       = 2.*np.pi*np.random.uniform()
    X        = R*np.cos(Th)
    Y        = R*np.sin(Th)
    Age      = np.random.uniform(0, 12)
    
    XRel     = X - 8.
    YRel     = Y
    ZRel     = Z

    RRel     = np.sqrt(XRel**2 + YRel**2 + ZRel**2)

    ResDict  = {'Age': Age, 'Xkpc': X, 'Ykpc': Y, 'Zkpc': Z, 'Rkpc': R, 'Th': Th, 'XRelkpc': XRel, 'YRelkpc':YRel, 'ZRelkpc': ZRel, 'RRelkpc': RRel}
    
    return ResDict

 
def Sample1DPop(NBin,Hr,Hz):
    RRandSet = np.random.uniform(0, 1, NBin)
    ZRandSet = np.random.uniform(0, 1, NBin)
    ZSignSet = np.sign(np.random.uniform(0, 2, NBin) - 1)
    RSet     = np.asarray([RCDFInv(RRandSet[i],Hr) for i in range(NBin)],dtype=float)
    ZSet     = np.asarray([zCDFInv(ZRandSet[i],Hz) for i in range(NBin)],dtype=float)*ZSignSet
    ThSet    = np.random.uniform(0, 2.*np.pi, NBin)
    XSet     = RSet*np.cos(ThSet)
    YSet     = RSet*np.sin(ThSet)
    AgeSet   = np.random.uniform(0, 12, NBin)
    IDSet    = np.arange(NBin) + 1    
    
    ResDict  = {'ID':IDSet, 'Ages': AgeSet, 'Xkpc': XSet, 'Ykpc': YSet, 'Zkpc': ZSet, 'Rkpc': RSet, 'Th': ThSet}
    ResDF    = pd.DataFrame(ResDict)    
    return ResDF

ExportTable = False

if ExportTable:
    NBin = 10**4
    Hr   = 4
    Hz   = 0.5
    
    Res = Sample1DPop(NBin,Hr,Hz)
    
    Res.to_csv('./GalTest.csv', index = False)
                
    
    
    
    