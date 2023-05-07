#!/home/amin/miniconda3/bin/python
import sys
import h5py
import pandas as pd
import numpy as np

def paired_sort(df, images, column):
    df.sort_values(column, inplace=True)
    images = images[list(df.index),...]
    df.reset_index(inplace=True, drop=True)
    return df, images
    

if __name__ == '__main__':
    file_name = sys.argv[1]
    column = sys.argv[2]
    df = pd.read_hdf(file_name, mode='r', key='features')
    with h5py.File(file_name, mode='r') as file:
        images = file['images'][:]

    df, images = paired_sort(df, images, column)
    
    with h5py.File(file_name, mode='r+') as file:
        file['images'][:] = images
    
    df.to_hdf(file_name, mode='r+', key='features')