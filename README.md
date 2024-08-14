# Proton Computed Tomography (ASU & Mayo Clinic)

## Participants 
Ricardo Alarcon: Professor at ASU  
Skylar Clymer: Physics PhD Student at ASU  
Evrim Gulser: Physics Undergrad at ASU  
Lukas Domer: Physics Undergrad at ASU  
Arda Gulser: Physics Undergrad at ASU 
Paul Mulqueen: CEO of Proton Calibration Technologies  
Stephen Sledge: Proton Calibration Technologies   
Daniel Robertson: Mayo Clinic  
Martin Bues: Mayo Clinic  

## Goal of PCT
The goal of the PCT efforts is to provide a strong foundation and proof-of-concept for the ability to use therapeutic proton beams for the purpose of medical imaging, particularly to assist in more accurate imaging for cancer therapy.

## How it works
Protons exit the accelerator at some initial energy $E_0$. After passing through a target, the protons lose some amount of energy based on the material that they pass through.

$\Delta E = E - E_0$

If we can measure the energy loss, we can start putting together what material they might have passed through.

A thin cesium iodide scintillating crystal is placed beyond the target. The crystal will emit light as a function of the number of protons as well as their energies. So if we shoot an 80MeV beam through a target, and the crystal emits an amount of light that is consisten with 63MeV protons, then we can reason that the energy lost in the target was 17MeV, so we have a way to calculate energy loss.

## Data
We capture a proportion of the emitted light with a standard digital camera. We take the raw, unprocessed file from that and analyze these files. This approach basically turns the pixels into binned photon counters, where each pixel value is some scalar multiple of the number of photons that hit the pixel.

Pixel value = $\alpha \cdot N$

where N is the number of photons that hit the pixel. We do not know the value of $\alpha$ and it may not matter. This may still be up for change.

## Current State of the Experiment
We have mostly, but not fully, solidified the analysis methods which will be documented in more depth at a later date. Most functions that relate to this analysis are found in the ARW_Support_2 file.

We are trying to verify just how accurately the data matches to the theory, and to gauge if there's enough information to make predictions of the exit energy.

## Files 
### test.py 
Just my personal scratch work. There is no designed order, documentation, or purpose. It's just for me to do things that (hopefully) only need to be done once. This is mostly for me. I can't imagine this file being of use for anyone else.

### Observer Notes 
Notes taken by Arda Gulser of the ASU team that detail the specifics of each run and log any changes made over the course of the days that data was taken. 

### Find_Median_Runs.py 
Our camera was not synced to the beam, leading to varying amounts of the light being captured. Some were taken with beam off, some were taken while beam was turning on/off, and some captured the beam during the entire time the shutter was open. This file finds the median brightness, where "brightness" is generally just meaning the total sum of all pixel values are we process them using predetermined methods.

### Database.json
A dictionary of layered dictionaries of preprocessed data for all files that we've determined to be "useful".

### Database Creator.py 
The file that generates Database. Database has had some changes since this was originally run so some issues that were prevalent upon creation have been fixed.

### ARW_Support 
ARW_Support.py was the original support module I made to assist in making the analysis efforts more user-friendly. Upon many feature requests, I began to realize the design was getting sloppy and hard to maintain. It has since been replaced but will be kept around for historical records.

ARW_Support_2.py is the updated version. It has a number of functions removed from the previous version that were no longer used. Many functions have been generalized more and split up to keep things more maintainable, more customizable for the user, and better documented. This is the one that is currently being used for analysis.
