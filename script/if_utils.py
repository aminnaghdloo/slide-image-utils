
# Copyright (C) December 2021 
#
# Author: Amin Naghdloo
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# ====================================================================

### Note:

# The following package is capable of processing nucleated cells.
# It will later be developed to include non-nucleated events as well.


# Loading packages
import glob
import os
import sys
import math
import pickle
import cv2
import numpy as np
import pandas as pd
import tifffile as tff
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util, exposure
)

## Parameters
# General
img_pref = "Tile"
img_suf = ".tif"
channels = {'DAPI' : 0, 'TRITC' : 1, 'CY5' : 2, 'FITC' : 4}
n_frames = 2304

# data_path is where any output will be saved.
# data_path = "/home/amin/Dropbox/Education/PhD/CSI/Projects/P13_immune/data"
data_path = "/media/amin/B47805FB7805BCDC/data_path"
onco_paths = [f"/media/{item}/OncoScope" for item in ['T', 'V', 'W', 'Z', 'R']]
dz_path = "/media/Y/DZ"

# Classes
class Frame:
    """A class to include 10x frame image data"""
    def __init__(self, slide_id, frame_id, data_path=data_path):
        self.slide_id = slide_id
        self.frame_id = frame_id
        self.image = getFrameImage(slide_id, frame_id)
        self.nucleusmask = self.readNucleusMask()
        self.cellmask = self.readCellMask()
        self.count = np.unique(self.nucleusmask).shape[0] - 1
    
    def readNucleusMask(self):
        nucleusmask_dir = f"{data_path}/{self.slide_id}/nucleusmask"
        os.makedirs(nucleusmask_dir, exist_ok=True)
        nucleusmask_path = f"{nucleusmask_dir}/{self.frame_id}.tif"
        if os.path.exists(nucleusmask_path):
            return(readMask(nucleusmask_path))
        else:
            print(f"Generating nucleus mask -> {self.slide_id} :"
                f" {self.frame_id}")
            return(segmentNucleus(self.image[:, :, 0], nucleusmask_path))

    def readCellMask(self):
        cellmask_dir = f"{data_path}/{self.slide_id}/cellmask"
        os.makedirs(cellmask_dir, exist_ok=True)
        cellmask_path = f"{cellmask_dir}/{self.frame_id}.tif"
        if os.path.exists(cellmask_path):
            return(readMask(cellmask_path))
        else:
            print(f"Generating cell mask -> {self.slide_id} :"
                f" {self.frame_id}")
            return(segmentCell(self.image, self.nucleusmask, cellmask_path))

    def extractCellCoords(self):
        "Extract cell coordinates of the frame based on the mask."
        props = measure.regionprops_table(
                    self.cellmask, self.image, separator='_',
                    properties=['label', 'centroid']
                    )
        props = pd.DataFrame(props)
        props.set_axis(['cell_id', 'x', 'y'], axis=1, inplace=True)
        props.insert(0, 'slide_id', self.slide_id)
        props.insert(1, 'frame_id', self.frame_id)
        return(props)
    
    def extractCellProps(self):
        "Extract area, eccentricity, and intensity features"
        props1 = measure.regionprops_table(
                    self.nucleusmask, self.image[:, :, 0], separator='_',
                    properties=['area', 'eccentricity', 'intensity_mean']
                    )
        props1 = pd.DataFrame(props1)
        props1.set_axis(['nucleus_area', 'nucleus_eccentricity',
                         f"{list(channels.keys())[0]}_mean_intensity"],
                        axis=1, inplace=True
                       )
        props2 = measure.regionprops_table(
                    self.cellmask, self.image[:, :, 1:], separator='_',
                    properties=['area', 'eccentricity', 'intensity_mean']
                    )
        props2 = pd.DataFrame(props2)
        names = ['cell_area', 'cell_eccentricity']
        names.extend([f"{ch}_mean_intensity"
                      for ch in list(channels.keys())[1:]])
        props2.set_axis(names, axis=1, inplace=True)
        props = pd.concat([props1, props2], axis=1)
        return(props)
    
    def extractGranularity(self, n=10, channel=0, use_nucleus_mask=True):
        "Extract granularity features from the given channel."
        if(use_nucleus_mask):
            df = measure.regionprops(self.nucleusmask, self.image[:,:,channel])
        else:
            df = measure.regionprops(self.cellmask, self.image[:,:,channel])

        all_features = []
        for i in range(len(df)):
            test_image = df[i].image_intensity
            mask = df[i].image
            image = exposure.rescale_intensity(
                test_image, in_range=(test_image.min(), test_image.max()),
                out_range=(0, 255))
            image = image.astype('uint8')
            pixels = image.copy()
            startmean = np.mean(pixels[mask])
            ero = pixels.copy()
            ero[~mask.astype('bool')] = 0
            currentmean = startmean
            footprint = morphology.disk(1, dtype=bool)

            object_features = []
            for j in range(n):
                prevmean = currentmean
                ero_mask = np.zeros_like(ero)
                ero_mask[mask == True] = ero[mask == True]
                ero = morphology.erosion(ero_mask,
                                         footprint=footprint)
                rec = morphology.reconstruction(ero, pixels,
                                                footprint=footprint)
                currentmean = np.mean(rec[mask])
                gs = (prevmean - currentmean) * 100 / startmean
                object_features.append(gs)

            all_features.append(object_features)
        prefix = list(channels.keys())[channel]
        out = pd.DataFrame(all_features, 
                           columns=[f"{prefix}_gran_{i}"
                                    for i in range(1,n+1)])
        return(out)


class Frame40x():
    "A class to include all 40x frame image data."

    def __init__(self, slide_id, frame_id):
        self.slide_id = slide_id
        self.frame_id = frame_id
        self.image = get40xFrameImage(slide_id, frame_id)
        self.nucleusmask = self.readNucleusMask()
        self.cellmask = self.readCellMask()
        self.count = np.unique(self.nucleusmask).shape[0] - 1
    
    def readNucleusMask(self):
        nucleusmask_dir = f"{data_path}/{self.slide_id}/40X/nucleusmask"
        os.makedirs(nucleusmask_dir, exist_ok=True)
        nucleusmask_path = f"{nucleusmask_dir}/{self.frame_id}.tif"
        if os.path.exists(nucleusmask_path):
            return(readMask(nucleusmask_path))
        else:
            print(f'''Generating nucleus mask -> 
            {self.slide_id} : {self.frame_id}''')
            return(segment40xNucleus(self.image[:, :, 0], nucleusmask_path))

    def readCellMask(self):
        cellmask_dir = f"{data_path}/{self.slide_id}/40X/cellmask"
        os.makedirs(cellmask_dir, exist_ok=True)
        cellmask_path = f"{cellmask_dir}/{self.frame_id}.tif"
        if os.path.exists(cellmask_path):
            return(readMask(cellmask_path))
        else:
            print(f"Generating cell mask -> {self.slide_id} : {self.frame_id}")
            return(segment40xCell(self.image, self.nucleusmask, cellmask_path))
        

# Functions
def getSlidePath(slide_id):
    "Extracts path to raw 10x frame images of a slide."
    for onco_path in onco_paths:
        slide_path = glob.glob(f"{onco_path}/"
                           f"tubeID_{slide_id[0:5]}/*/slideID_{slide_id}/"
                           f"bzScanner/proc/")
        if len(slide_path) != 0:
            return(slide_path[0])
    print("Slide path was not found!")
    return(None)


def getFrameImage(slide_id, frame_id):
    "Extracts raw 10x frame image of a slide."
    slide_path = getSlidePath(slide_id)
    image_names = [f"{slide_path}{img_pref}"
                   f"{str(frame_id + n_frames * id).zfill(6)}{img_suf}"
                   for id in channels.values()]
    for name in image_names:
        assert os.path.exists(name)
    out_image = np.stack(
        [cv2.imread(image_names[i],-1) for i in range(len(channels))],
        axis=2
        )
    return(out_image)


def readMask(mask_path):
    "Read masks using OpenCV library with 16-bit integer values."
    out = cv2.imread(mask_path, -1)
    return(out)


def cvImageShow(img, title='title'):
    "Display image using OpenCV library."
    cv2.imshow('sample image',img)
    cv2.waitKey(3000) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image


def pltImageShow(img, title='title', size=(8,6), dpi=150):
    "Display image using matplotlib.pyplot package."
    if(img.dtype == 'uint16'):
        img = img.astype('float')
        img = img / 65535
    plt.figure(figsize=size,dpi=dpi)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(title)
    plt.show()


def fillHull(input_mask):
    "Fill the holes in masks."
    assert input_mask.dtype == np.uint8
    h, w = input_mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:h + 1, 1:w + 1] = input_mask.copy()
    canvas[canvas != 0] = 1
    mask = np.zeros((h + 4, w + 4), np.uint8)
    mask[1:h + 3, 1:w + 3] = canvas
    cv2.floodFill(canvas, mask, (0, 0), 1)
    canvas = 1 - canvas[1:h + 1, 1:w + 1]
    return(canvas * input_mask.max() + input_mask)


def segmentNucleus(input_image, file_name=None):
    '''Segments cell nuclei from DAPI channel grayscale image and returns 
    a grayscale image with unique labels for event masks.
    The mask can also be saved provided the 'file_name' argument.'''
    
    # Segmentation parameters
    blur_size = (5, 5)
    thresh_size = 25
    thresh_offset = -2 # threshold offset for adaptive thresholding
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

    # Image processing
    image = cv2.blur(input_image, blur_size) # Mean filter to reduce noise
    cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    image = image.astype("uint8")
    mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, thresh_size, thresh_offset)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)
    mask = fillHull(mask) 
    image_dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3, cv2.CV_32F)
    local_max_coords = feature.peak_local_max(image_dist, min_distance=7)
    local_max_mask = np.zeros(image_dist.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = measure.label(local_max_mask)
    segmented_nuclei = segmentation.watershed(-image_dist, markers, mask=mask)
    segmented_nuclei = segmented_nuclei.astype("uint16")

    if file_name is not None:
        cv2.imwrite(file_name, segmented_nuclei)

    return(segmented_nuclei)


def segmentCell(image, seed, file_name=None):
    '''Segments cells based on markers using all given grayscale channels in 
    the 'input_image' and returns a grayscale image with unique labels for 
    event masks.
    The mask can also be saved provided the file_name argument.'''
    
    # Segmentation parameters
    blur_size = (5, 5)
    thresh_size = 25
    thresh_offset = -2 # threshold offset for adaptive thresholding
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    
    image = image.astype("float64")
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    for i in range(image.shape[2]):
        image[:, :, i] = cv2.blur(image[:, :, i], blur_size)
        image[:, :, i] = cv2.normalize(image[:, :, i], 0, 65535,
                                       cv2.NORM_MINMAX)
    
    out_image = np.zeros(image.shape[0:2], dtype='float64')
    for i in range(image.shape[2]):
        out_image += image[:,:,i] ** 2
    out_image = np.sqrt(out_image / image.shape[2])
        
    cv2.normalize(out_image, out_image, 0, 255, cv2.NORM_MINMAX)
    out_image = out_image.astype('uint8')
    mask = cv2.adaptiveThreshold(
        out_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        thresh_size, thresh_offset
    )
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)
    mask = fillHull(mask)
    image_dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3, cv2.CV_32F)
    segmented_cells = segmentation.watershed(-image_dist, seed, mask=mask)
    segmented_cells = segmented_cells.astype("uint16")
    
    if file_name is not None:
        cv2.imwrite(file_name, segmented_cells)
        
    return(segmented_cells)


def segment40xNucleus(input_image, file_name=None):
    '''Segments cell nuclei from DAPI channel grayscale image and returns 
    a grayscale image with labeled masks.
    The mask can also be saved provided the file_name argument.'''
    
    # Segmentation Parameters
    blur_size = (15, 15)
    thresh_size = 55
    thresh_offset = -1 # threshold offset for adaptive thresholding
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

    # Image processing
    image = cv2.blur(input_image, blur_size) # Mean filter to reduce noise
    cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    image = image.astype("uint8")
    mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, thresh_size, thresh_offset)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)
    mask = fillHull(mask)
    image_dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3, cv2.CV_32F)
    local_max_coords = feature.peak_local_max(image_dist, min_distance=15)
    local_max_mask = np.zeros(image_dist.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = measure.label(local_max_mask)
    segmented_nuclei = segmentation.watershed(-image_dist, markers, mask=mask)
    segmented_nuclei = segmented_nuclei.astype("uint16")

    if file_name is not None:
        cv2.imwrite(file_name, segmented_nuclei)

    return(segmented_nuclei)


def segment40xCell(image, seed, file_name=None):
    '''Segments cells based on markers using all given grayscale channels in 
    the 'input_image' and returns a grayscale image with unique labels for 
    event masks.
    The mask can also be saved provided the file_name argument.'''
    
    # Segmentation parameters
    blur_size = (15, 15)
    thresh_size = 99
    thresh_offset = -1 # threshold offset for adaptive thresholding
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    
    image = image.astype("float64")
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    for i in range(image.shape[2]):
        image[:, :, i] = cv2.blur(image[:, :, i], blur_size)
        image[:, :, i] = cv2.normalize(image[:, :, i], 0, 65535,
                                       cv2.NORM_MINMAX)
    
    out_image = np.zeros(image.shape[0:2], dtype='float64')
    for i in range(image.shape[2]):
        out_image += image[:,:,i] ** 2
    out_image = np.sqrt(out_image / image.shape[2])

    cv2.normalize(out_image, out_image, 0, 255, cv2.NORM_MINMAX)
    out_image = out_image.astype(np.uint8)
    mask = cv2.adaptiveThreshold(
        out_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        thresh_size, thresh_offset
    )
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)
    mask = fillHull(mask)
    image_dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3, cv2.CV_32F)
    segmented_cells = segmentation.watershed(-image_dist, seed, mask=mask)
    segmented_cells = segmented_cells.astype("uint16")
    
    if file_name is not None:
        cv2.imwrite(file_name, segmented_cells)
    
    return(segmented_cells)


def getCellCoords(input_file, output_file=None):
    input_df = pd.read_table(input_file)
    input_df.frame_id = input_df.frame_id.astype('int')
    groups = input_df.groupby(['slide_id', 'frame_id'])
    output_df = []
    for slide_id, frame_id in groups.indices:
        frame = Frame(slide_id, frame_id)
        temp = frame.extractCellCoords()
        output_df.append(temp)
        print(f"{frame.count} cell coordinates extracted from slide_id:"
              f"{slide_id}, frame_id:{frame_id}")
    output_df = pd.concat(output_df)
    
    if output_file is not None:
        output_df.to_csv(output_file, sep="\t", index=False)
        
    return(output_df)


def getCellfeatures(input_file, output_file=None):
    input_df = pd.read_table(input_file)
    input_df.frame_id = input_df.frame_id.astype('int')
    groups = input_df.groupby(['slide_id', 'frame_id'])
    output_df = []
    for slide_id, frame_id in groups.indices:
        frame = Frame(slide_id, frame_id)
        temp1 = frame.extractCellCoords()
        temp2 = frame.extractCellProps()
        temp3 = frame.extractGranularity(channel=0, use_nucleus_mask=True)
        temp4 = frame.extractGranularity(channel=2, use_nucleus_mask=False)
        temp = pd.concat([temp1, temp2, temp3, temp4], axis=1)
        output_df.append(temp)
        print(f"{frame.count} cell features extracted from slide_id:"
              f"{slide_id}, frame_id:{frame_id}")
    output_df = pd.concat(output_df)
    if output_file is not None:
        output_df.to_csv(output_file, sep="\t", index=False)
    return(output_df)


def getGranularity(input_file, output_file=None, n=10, channel=0,
                   use_nucleus_mask=False):
    input_df = pd.read_table(input_file)
    input_df.frame_id = input_df.frame_id.astype('int')
    groups = input_df.groupby(['slide_id', 'frame_id'])
    output_df = []
    for slide_id, frame_id in groups.indices:
        frame = Frame(slide_id, frame_id)
        temp = frame.extractGranularity(n=n, channel=channel,
                                        use_nucleus_mask=use_nucleus_mask)
        output_df.append(temp)
        print(f"{frame.count} cell granularity extracted from slide_id:"
              f"{slide_id}, frame_id:{frame_id}")
    output_df = pd.concat(output_df)
    if output_file is not None:
        output_df.to_csv(output_file, sep="\t", index=False)
    return(output_df)


def getCellImages(input_file, width, mask_flag=True):
    '''
    file_path: a tab-delimited list of cells including 4 columns:
    <slide_id>  <frame_id>  <x> <y>,
    width: cropping size of each individual cell image
    mask: flag to mask the individual cell of interest
    '''
    cell_info = pd.read_table(input_file)
    cell_info.frame_id = cell_info.frame_id.astype(int)
    images = np.zeros((cell_info.shape[0], width, width, len(channels)),
                    dtype='uint16')
    groups = cell_info.groupby(['slide_id', 'frame_id'])
    for slide_id, frame_id in groups.indices:
        frame = Frame(slide_id, frame_id)
        index = groups.indices[(slide_id, frame_id)]
        images[index.tolist(), :, :, :] = cropCells(
                        frame, cell_info.loc[index.tolist(), ['x','y']],
                        width, mask_flag
                        )
        print(f"{len(index)} cell images extracted from slide_id:{slide_id}"
              f", frame_id:{frame_id}")
    return(images)


def cropCells(frame, xy_data, width, mask_flag):
    edge = round((width - 1) / 2)
    xy_data.x = xy_data.x.astype(int) + edge
    xy_data.y = xy_data.y.astype(int) + edge
    pad_color = np.zeros(len(channels))
    mask = cv2.copyMakeBorder(frame.cellmask, edge, edge, edge, edge, 
                    cv2.BORDER_CONSTANT, 0)
    mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
    image = cv2.copyMakeBorder(frame.image, edge, edge, edge, edge,
                    cv2.BORDER_CONSTANT, pad_color)
    ids = mask[xy_data.x - 1, xy_data.y - 1].flatten()
    indices = xy_data.index.tolist()
    out = np.zeros((xy_data.shape[0], width, width, len(channels)),
                    dtype='uint16')
    for i,index,id in zip(range(xy_data.shape[0]), indices, ids):
        if(mask_flag):
            scimage = np.multiply(image, (mask == id).astype('uint16'))
        else:
            scimage = image
        out[i,:,:,:] = scimage[(xy_data.loc[index, 'x'] - edge):
                        (xy_data.loc[index, 'x'] + edge + 1),
                        (xy_data.loc[index, 'y'] - edge):
                        (xy_data.loc[index, 'y'] + edge + 1),:]
    
    return(out)


def fourchannels2rgb(image):
    "Convert 4 channel images to RGB float to display."
    assert(image.dtype == 'uint16')
    image = image.astype('float')
    if(len(image.shape) == 4):
        image[:,:,:,0:3] = image[:,:,:,[1,2,0]]
        image = image[:,:,:,0:3] + np.expand_dims(image[:,:,:,3], 3)
        
    elif(len(image.shape) == 3):
        image[:,:,0:3] = image[:,:,[1,2,0]]
        image = image[:,:,0:3] + np.expand_dims(image[:,:,3], 2)
        
    image[image > 65535] = 65535
    image = image.astype('uint16')
    return(image)


def createGallery(images, n_h, n_w):
    "Create 2D gallery from list of images."
    shape = images.shape
    h = shape[1] * n_h
    w = shape[2] * n_w
    d = shape[3]
    pages = math.ceil(len(images)/(n_h * n_w))
    n_total = pages * n_h * n_w
    n_current = len(images)
    if(n_total != n_current):
        filler = np.zeros((n_total - n_current, shape[1], shape[2], d),
                          dtype = images.dtype)
        images = np.append(images, filler, axis=0)
    gallery = np.zeros((pages, h, w, d), dtype='uint16')
    for k in range(pages):
        gallery[k, :, :, :] = np.vstack(
            [np.hstack(
                [images[(k * n_w * n_h) + (j * n_w) + i, :, :, :]
                     for i in range(n_w)]
                ) for j in range(n_h)]
            )
    return(gallery)


def saveImages(array, path):
    "Save a list of images as uint8 image file."
    if(array.shape[-1] == 3):
        array = np.flip(array, -1)
        
    if(len(array.shape) == 4):
        cv2.imwritemulti(path, array)
        print(f"Images saved in {path}")
    else:
        cv2.imwrite(path, array)
        print(f"Image saved in {path}")


def saveData(path, data):
    """Save any data in .pickle format."""
    file = open(path + '.pickle', 'wb')
    pickle.dump(data, file)
    file.close()
    print(f"Data saved at {path}.pickle!")
    
    
def loadData(path):
    """Load any data in .pickle format."""
    f = open(path, "rb")
    data = pickle.load(f)
    print(f"Data loaded from {path}!")
    return(data)


def saveCompressedData(path, data):
    "Save any data as bz2-compressed pickle file."
    with bz2.BZ2File(path + '.pbz2', 'w') as f:
        cPickle.dump(data, f)
        print(f"Data saved at {path}.pbz2!")

        
def loadCompressedData(path):
    "Load any bz2-compressed data from pickle file. (''.pbz2')"
    data = bz2.BZ2File(path, 'rb')
    data = cPickle.load(data)
    print(f"Data loaded from {path}!")
    return(data)


#def saveMontages()



## Functions for 40x image processing
def get40xSlidePath(slide_id):
    "Extracts path to raw 40x frame images of a slide."
    slide_path = glob.glob(f"{dz_path}/{slide_id}/**/40X")
    return(slide_path[0])


def get40xFrameImage(slide_id, frame_id):
    '''Extract raw 40x a frame image using slide_id and frame_id.
    
    A 40x image 'frame_id' has the following format:
    <frame_id>-<cell_id>-<x>-<y>
    where all for information correspond to the interesting cell 
    selected from 10x image processing.
    
    Example from slide 0A4801:
    1013-590-966-200 
    
    This function uses 'tifffile' module to read 12-bit images and 
    performs 4 left bit-shifts (multiplication by 16) to normalize 
    the image.'''
    
    slide_path = get40xSlidePath(slide_id)
    image_names = [f"{slide_path}/{frame_id}-{ch}.tif"
                   for ch in channels.keys()]
    out_image = np.stack(
        [tff.imread(image_names[i]) for i in range(len(channels))],
        axis=2
        )
    return(out_image * 16)


