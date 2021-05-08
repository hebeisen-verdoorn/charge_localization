#!/usr/bin/env python
# coding: utf-8

# !/usr/bin/env python
import os
import sys
import glob

from I3Tray import *
from icecube import dataclasses, dataio, simclasses, phys_services, linefit, DomTools, lilliput, paraboloid, cramer_rao
from icecube.icetray import I3Units

import numpy as np
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')



def deltapsi(zenith1, azimuth1, zenith2, azimuth2):
        return np.degrees(np.arccos(np.sin(zenith1)*np.sin(zenith2)*(np.sin(azimuth1)*np.sin(azimuth2)+np.cos(azimuth1)*np.cos(azimuth2))+np.cos(zenith1)*np.cos(zenith2)))

inpath = '/data/user/hebeisen-verdoon/charge_localization/11499/'

angular_deviation_SPE = []
angular_deviation_MPE = []
angular_deviation_localization = []
SPE_CramerRaoSigma = []
MPE_CramerRaoSigma = []
Localization_CramerRaoSigma = []
lever_arm = []
zenith_SPE = []
zenith_MPE = []
zenith_Localization = []
zenith_MC = []
azimuth_SPE = []
azimuth_MPE = []
azimuth_Localization = []
azimuth_MC = []
Qtot = []
brightest_dom_charge = []
second_brightest_dom_charge = []
N=0

for f in os.listdir(inpath):
    if f.endswith('.i3.zst'):
        infile = dataio.I3File(inpath+f)
        for frame in infile:
            if frame.Stop != icetray.I3Frame.Physics: continue
            header = frame['I3EventHeader']
            if header.sub_event_stream != 'InIceSplit': continue
            true_particle = frame['MMCTrackList'][0].particle
            true_zenith = true_particle.dir.zenith
            true_azimuth = true_particle.dir.azimuth
            SPE_zenith = frame['SPEFit2'].dir.zenith
            SPE_azimuth = frame['SPEFit2'].dir.azimuth
            SPE_sigma = np.sqrt((np.degrees(frame['SPEFit2CramerRaoParamsParams'].variance_theta) +
                                         np.degrees(frame['SPEFit2CramerRaoParamsParams'].variance_phi))/2)
            MPE_zenith = frame['MPEFit'].dir.zenith
            MPE_azimuth = frame['MPEFit'].dir.azimuth
            MPE_sigma = np.sqrt((np.degrees(frame['MPEFitCramerRaoParams'].variance_theta) +
                                         np.degrees(frame['MPEFitCramerRaoParams'].variance_phi))/2)
            Localization_zenith = frame['localization_line'].dir.zenith
            Localization_azimuth = frame['localization_line'].dir.azimuth
            Localization_sigma = np.sqrt((np.degrees(frame['CramerRao_localizationParams'].variance_theta) +
                                         np.degrees(frame['CramerRao_localizationParams'].variance_phi))/2)
            l = frame['localization_line'].length
            q = frame['CVStatistics'].q_tot_pulses
            q1 = frame['brightest_dom_charge'].value
            q2 = frame['second_brightest_dom_charge'].value
            if q < 30: continue
            if l < 200: continue
            if (Localization_sigma > 180) or (SPE_sigma > 180) or (MPE_sigma > 180): continue
            if q1 < 0.1: continue
            if q2 < 0.075: continue
            if q1+q2 < 0.2: continue
            brightest_dom_charge.append(q1)
            second_brightest_dom_charge.append(q2)
            Qtot.append(q)
            lever_arm.append(l)
            zenith_SPE.append(SPE_zenith)
            zenith_MPE.append(MPE_zenith)
            zenith_Localization.append(Localization_zenith)
            zenith_MC.append(true_zenith)
            azimuth_SPE.append(SPE_azimuth)
            azimuth_MPE.append(MPE_azimuth)
            azimuth_Localization.append(Localization_azimuth)
            azimuth_MC.append(true_azimuth)
            SPE_CramerRaoSigma.append(SPE_sigma)
            MPE_CramerRaoSigma.append(MPE_sigma)
            Localization_CramerRaoSigma.append(Localization_sigma)
            deltapsi_SPE = deltapsi(SPE_zenith, SPE_azimuth, true_zenith, true_azimuth)
            deltapsi_MPE = deltapsi(MPE_zenith, MPE_azimuth, true_zenith, true_azimuth)
            deltapsi_Localization = deltapsi(Localization_zenith, Localization_azimuth, true_zenith, true_azimuth)
            angular_deviation_SPE.append(deltapsi_SPE)
            angular_deviation_MPE.append(deltapsi_MPE)
            angular_deviation_localization.append(deltapsi_Localization)
            N=N+1

logbins = np.logspace(-1, np.log10(180), 50)

angular_deviation_SPE_median = np.nanmedian(angular_deviation_SPE)
angular_deviation_MPE_median = np.nanmedian(angular_deviation_MPE)
angular_deviation_localization_median = np.nanmedian(angular_deviation_localization)

plt.hist(angular_deviation_SPE, histtype='step', bins=logbins, label='SPE')
plt.hist(angular_deviation_MPE, histtype='step', bins=logbins, label='MPE')
plt.hist(angular_deviation_localization, histtype='step', bins=logbins, label='Localization')
plt.vlines(angular_deviation_SPE_median, ymin=0, ymax=200000, color='#1f77b4')
plt.vlines(angular_deviation_MPE_median, ymin=0, ymax=200000, color='#ff7f0e')
plt.vlines(angular_deviation_localization_median, ymin=0, ymax=200000, color='#2ca02c')
plt.ylim(0,30000)
plt.legend(loc='upper left')
plt.title('Angular Deviation')
plt.xscale('log')
#plt.xticks([0.1, 1, 10, 100], labels=['0.1', '1', '10', '100'])
plt.xlim(0.1,180)
plt.xlabel('\u0394 \u03C8 [deg]')
plt.ylabel('Events')
plt.show()

from scipy.stats import median_absolute_deviation

pull_SPE = np.subtract(np.degrees(zenith_SPE), np.degrees(zenith_MC))
pull_MPE = np.subtract(np.degrees(zenith_MPE), np.degrees(zenith_MC))
pull_Localization = np.subtract(np.degrees(zenith_Localization), np.degrees(zenith_MC))
pull_SPE_median = np.nanmedian(pull_SPE)
pull_MPE_median = np.nanmedian(pull_MPE)
pull_Localization_median = np.nanmedian(pull_Localization)
pull_SPE_mad = median_absolute_deviation(pull_SPE, nan_policy='omit')
pull_MPE_mad = median_absolute_deviation(pull_MPE, nan_policy='omit')
pull_Localization_mad = median_absolute_deviation(pull_Localization, nan_policy='omit')
print('Medians:', pull_SPE_median, pull_MPE_median, pull_Localization_median)
print('MAD:', pull_SPE_mad, pull_MPE_mad, pull_Localization_mad)

fig, axes = plt.subplots(1, 3, figsize=[19.2, 4.8])
axes[0].hist(pull_SPE[~np.isnan(pull_SPE)], bins=100, histtype='step')
axes[0].set_title('SPE')
axes[0].set_xlim(-50,50)
axes[1].hist(pull_MPE[~np.isnan(pull_MPE)], bins=100, histtype='step')
axes[1].set_title('MPE')
axes[1].set_xlim(-50,50)
axes[2].hist(pull_Localization[~np.isnan(pull_Localization)], bins=100, histtype='step')
axes[2].set_title('Localization')
axes[2].set_xlim(-50,50)
plt.show()

from scipy.stats import median_absolute_deviation, norm
from scipy.optimize import curve_fit

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

delta_zenith_SPE = np.subtract(np.degrees(zenith_Localization), np.degrees(zenith_SPE))
delta_zenith_MPE = np.subtract(np.degrees(zenith_Localization), np.degrees(zenith_MPE))

delta_zenith_SPE_median = np.nanmedian(delta_zenith_SPE)
delta_zenith_MPE_median = np.nanmedian(delta_zenith_MPE)
delta_zenith_SPE_mad = median_absolute_deviation(delta_zenith_SPE, nan_policy='omit')
delta_zenith_MPE_mad = median_absolute_deviation(delta_zenith_MPE, nan_policy='omit')
print('Medians:', delta_zenith_SPE_median, delta_zenith_MPE_median)
print('MAD:', delta_zenith_SPE_mad, delta_zenith_MPE_mad)

heights_SPE, bins_SPE, _ = plt.hist(delta_zenith_SPE[~np.isnan(delta_zenith_SPE)], bins=1000, histtype='step')
bins_SPE = bins_SPE[:-1] + np.diff(bins_SPE) / 2
popt_SPE, pcov_SPE = curve_fit(gaus,bins_SPE,heights_SPE,p0=[1,0,4])

heights_MPE, bins_MPE, _ = plt.hist(delta_zenith_MPE[~np.isnan(delta_zenith_MPE)], bins=1000, histtype='step')
bins_MPE = bins_MPE[:-1] + np.diff(bins_MPE) / 2
popt_MPE, pcov_MPE = curve_fit(gaus,bins_MPE,heights_MPE,p0=[1,0,4])

fig, axes = plt.subplots(1, 2, figsize=[12.8, 4.8])
axes[0].hist(delta_zenith_SPE[~np.isnan(delta_zenith_SPE)], bins=1000, histtype='step')
axes[0].plot(bins_SPE, gaus(bins_SPE, *popt_SPE))
axes[0].set_title('SPE')
axes[0].set_xlim(-50,50)
axes[1].hist(delta_zenith_MPE[~np.isnan(delta_zenith_MPE)], bins=1000, histtype='step')
axes[1].plot(bins_MPE, gaus(bins_MPE, *popt_MPE))
axes[1].set_title('MPE')
axes[1].set_xlim(-50,50)
plt.show()