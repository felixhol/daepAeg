'''
author: Felix Hol
date: 2020 Aug
content: code to process annotation output from CVAT to be used for U-net training.
Output: for each input image (CVAT SegmentationObject) 3 masks are saved:
1. eggs (not eroded)
2. edges of eggs
3. eroded eggs + edges
'''

import numpy as np
import os
import pims
import PIL
import skimage
from skimage import data, io, util
from skimage.measure import label, regionprops
from skimage.morphology import erosion, disk
import glob

#### set directories where to get images and where to store output, and specifics of experiment

dataDir = '/Users/felix/Documents/mosquitoes/mosquitoData/eggs/labeld/task_eggl5/SegmentationObject/'
saveDir = '/Users/felix/Documents/mosquitoes/mosquitoData/eggs/eggL/'
imgs = sorted( glob.glob(dataDir + '*.png'))
frames = pims.ImageSequence(dataDir+'/*.png', as_grey=True)


imgN = 0

for f in frames:
    mask = np.zeros(np.shape(f))
    erMask = np.zeros(np.shape(f))
    eggR = skimage.measure.regionprops(f)
    for egg in eggR:
        F = np.zeros(np.shape(f))
        F[egg.coords.T[0], egg.coords.T[1]] = 1
        mask = mask + F
        Fer = erosion(F,disk(1))
        erMask = erMask + Fer
    maskS = PIL.Image.fromarray((mask * 255).astype(np.uint8))
    maskS.save(saveDir + 'eggs' + os.path.basename(frames._filepaths[imgN]))
    edges = mask - erMask
    edges2 = edges
    edges2[edges2>0] = 2
    edgesS = PIL.Image.fromarray((edges * 255).astype(np.uint8))
    edgesS.save(saveDir + 'edges' + os.path.basename(frames._filepaths[imgN]))
    fullMask = edges2 + erMask
    fullMaskS = PIL.Image.fromarray((fullMask * 125).astype(np.uint8))
    fullMaskS.save(saveDir + 'eggsEdges' + os.path.basename(frames._filepaths[imgN]))
    imgN += 1



