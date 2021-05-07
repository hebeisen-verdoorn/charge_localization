#!/usr/bin/env python
# coding: utf-8

import os
import sys
import glob

from I3Tray import *
from icecube import dataclasses, dataio, simclasses, phys_services, linefit, DomTools, lilliput, paraboloid, cramer_rao # icecube software

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

geo = dataio.I3File('/mnt/c/Users/zverd/IceCube/calibration/GeoCalibDetectorStatus_2016.57531_V0.i3.gz','r') ## import GCD file
g = geo.pop_frame()
DOMcoord = np.full((87,61,3), np.nan) ## 86 61x3 arrays, [x,y,z] coordinates of each DOM, one array for each string
DOMcoord[1:,0] = np.inf ## strings and layers begin at index 1
for string in range(1,87):
    for layer in range(1,61):
        DOMcoord[string,layer] = np.array(g['I3Geometry'].omgeo[icetray.OMKey(string,layer,0)].position)

def localize(array): # used by charge_localization function, return array sorted by highest % total charge to lowest by DOM
        total_charge = np.sum(array[:,2]) ## total charge of the event
        percentage_charge = np.array([[x/total_charge] for x in array[:,2]]) ## percentage of total event charge each DOM received
        localized_array = np.hstack((array[:,[0,1]], percentage_charge)) 
        return localized_array[localized_array[:,2].argsort()][::-1], total_charge 

def angles(array): # convert [x,y,z] direction vector to astronomy oriented zenith and azimuth angles
    zenith = np.arccos(array[2])
    if array[1] > 0:
        azimuth = np.arccos(array[0]/np.sin(zenith))
    else:
        azimuth = 2*np.pi - np.arccos(array[0]/np.sin(zenith))
    return zenith, azimuth
    
def charge_localization(frame, pulses=''): # the algorithm that computes the reconstructed muon track. Can be used in this form as a module in icetray script.
    pulse_series = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, pulses)
    DOMcharge = [] ## getting the charge deposited in each DOM
    for omkey, pulses in pulse_series:
        DOMcharge.append(np.array([omkey.string, omkey.om, sum([p.charge for p in pulses])]))
    DOMcharge = np.asarray(DOMcharge)
    localized_charge, total_event_charge = localize(DOMcharge) ## returns [string, om, % total event charge], total charge of the event
    brightest_dom_coord = DOMcoord[int(localized_charge[0][0]), int(localized_charge[0][1])] ## coordintates of the brightest DOM
    brightest_dom_charge = localized_charge[0][2]
    second_brightest_dom_charge = 0
    for dom in localized_charge:
        if int(localized_charge[0][0]) != int(dom[0]): # if the second brightest DOM is on the same string as the brightest, add its charge to the brightest, move on to next brightest DOM
            second_brightest_dom_coord = DOMcoord[int(dom[0]), int(dom[1])]
            second_brightest_dom_charge = dom[2]
            break
        else:
            second_brightest_dom_coord = DOMcoord[int(dom[0]), int(dom[1])]
            brightest_dom_charge =+ dom[2]
    brightest_dom_distance = np.linalg.norm(brightest_dom_coord - second_brightest_dom_coord) ## distance betweeen the two brightest DOMs, called the lever arm
    if brightest_dom_coord[2] > second_brightest_dom_coord[2]:
        charge_localization_vector = (brightest_dom_coord - second_brightest_dom_coord)/brightest_dom_distance
    else:
        charge_localization_vector = (second_brightest_dom_coord - brightest_dom_coord)/brightest_dom_distance
    localization_zenith, localization_azimuth = angles(charge_localization_vector)
    particle = dataclasses.I3Particle() # creating an I3Particle object containing direction and length information
    particle.dir = dataclasses.I3Direction(-charge_localization_vector[0], -charge_localization_vector[1], -charge_localization_vector[2])
    particle.pos = dataclasses.I3Position(brightest_dom_coord[0], brightest_dom_coord[1], brightest_dom_coord[2])
    frame['localization_line'] = particle # writing I3Particle object to the I3File
    particle.length = brightest_dom_distance
    frame['brightest_dom_charge'] = dataclasses.I3Double(brightest_dom_charge) # writing brightest and second brightest DOM charge to I3File
    frame['second_brightest_dom_charge'] = dataclasses.I3Double(second_brightest_dom_charge)

inpath = 'directory containing data files'
outpath = 'write the output files here'
gcdfile = ['file path of GCD file for the run']

for f in os.listdir(inpath):
    infile = [inpath+f]
    outfile = outpath+f.replace('.i3.zst', '_localized.i3.zst')
    files = gcdfile+infile
    tray = I3Tray()
    tray.Add('I3Reader', 'Reader', FileNameList = files)
    tray.Add(charge_localization, pulses='SRTCleaningPulses') # this example assumes I3File has undergone SeededRT pulse cleaning. If not can use other pulses
    tray.Add("CramerRao","cramer_rao_localization", # computing Cramer-Rao parameters for the localization line
        InputResponse = 'SRTCleaningPulses', 
        InputTrack = 'localization_line', 
        OutputResult = "CramerRao_localization", AllHits = True)
    tray.AddModule( "I3Writer", "Writer", FileName = outfile )
    tray.Execute()
    tray.Finish()