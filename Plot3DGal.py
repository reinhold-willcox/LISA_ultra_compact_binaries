#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:05:24 2024

@author: alexey
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import legwork.source as source
import legwork.visualisation as vis
import astropy.units as u
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(1, './PyModules/')


# Read in the Galaxy data
df = pd.read_csv('./GalTest.csv')
# Extracting the columns
x = df['Xkpc']
y = df['Ykpc']
z = df['Zkpc']

# Create a new figure for the 3D plot with high DPI
fig = plt.figure(dpi=300)
ax = fig.add_subplot(111, projection='3d', facecolor='black')

# Plotting the points
ax.scatter(x, y, z, color='lime', s=1, alpha=0.05)  # 's' is the size of the points

# Set the background color
ax.set_facecolor('black')

# Set the range for all axes
ax.set_xlim([-15, 15])
ax.set_ylim([-15, 15])
ax.set_zlim([-15, 15])

# Set the labels and title with colors
ax.set_xlabel('X, kpc', color='white')
ax.set_ylabel('Y, kpc', color='white')
ax.set_zlabel('Z, kpc', color='white')

# Change the color of the tick labels to white
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')

# Save the plot as a high-resolution image
plt.savefig('./LISAGalTest.png', dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='black')


#Draw the DWDs
DWDArrPre = pd.read_csv('./DWD.csv')
#Assuming constant SFR from 14 Gyr till today, re-weight the distribution of DWDs
CurrAge    = 14000
NDraw      = 150
DWDUseList = []
while CurrAge>=500:
    #print(CurrAge)
    DWDCurrBin    = DWDArrPre[((DWDArrPre['time']<=CurrAge) & (DWDArrPre['time']>(CurrAge-500)))]
    #print(len(DWDCurrBin.index))
    DWDCurrBinUse = DWDCurrBin.sample(n=NDraw,random_state=1) 
    DWDUseList.append(DWDCurrBinUse)
    CurrAge -=500

DWDUse = pd.concat(DWDUseList)


DWDUse.to_csv('./DWDUseTest.csv',index=False)

NSys  = len(DWDUse.index)
GalDF = df.sample(n=NSys,random_state=1) 
GalDF['RRelSunKpc'] = np.sqrt((GalDF['Xkpc']-8)**2+ (GalDF['Ykpc'])**2 + (GalDF['Zkpc'])**2)

DWDUse = DWDUse.reset_index(drop=True)
GalDF = GalDF.reset_index(drop=True)

GalDWD = pd.concat([DWDUse,GalDF],axis=1)
    
GalDWD.to_csv('./GalDWDTest.csv',index=False)

#LEGWORK part

n_values = NSys
m_1 = (GalDWD['mass1']).to_numpy() * u.Msun
m_2 = (GalDWD['mass2']).to_numpy() * u.Msun
dist = (GalDWD['RRelSunKpc']).to_numpy() * u.kpc
f_orb = (1/(GalDWD['porb']*24.*60*60)).to_numpy() * u.Hz
ecc   = np.zeros(NSys)

sources = source.Source(m_1=m_1, m_2=m_2, ecc=ecc, dist=dist, f_orb=f_orb)
snr = sources.get_snr(verbose=True)

cutoff = -0.2

detectable_threshold = cutoff
detectable_sources = sources.snr > cutoff
print("{} of the {} sources are detectable".format(len(sources.snr[detectable_sources]), n_values))

# create the same plot but set `show=False`
fig, ax = sources.plot_source_variables(xstr="f_orb", ystr="snr", disttype="kde", log_scale=(True, True),
                                        fill=True, show=False, which_sources=sources.snr > 0, figsize=(10, 8))

#ax.set_aspect(0.8)

# duplicate the x axis and plot the LISA sensitivity curve
right_ax = ax.twinx()
frequency_range = np.logspace(np.log10(2e-6), np.log10(2e-1), 1000) * u.Hz
vis.plot_sensitivity_curve(frequency_range=frequency_range, fig=fig, ax=right_ax,fill=False)

plt.savefig('./SensitivityCurve.png')

