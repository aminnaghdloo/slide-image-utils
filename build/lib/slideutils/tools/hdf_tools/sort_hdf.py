#!/home/amin/miniconda3/bin/python
import sys
import h5py
import pandas as pd
import numpy as np

def paired_sort(df, images, masks, column, order_char):
    "A function that sorts images and masks "
    df.sort_values(
        column, inplace=True, ascending=True if order_char == 'A' else False)
    images = images[list(df.index),...]
    masks = masks[list(df.index),...] if masks is not None else None

    df.reset_index(inplace=True, drop=True)
    return df, images, masks
    

def main():

    if len(sys.argv) != 5:
        quit('python sort_hdf <input.hdf5> <output.hdf5> <column_name> <A/D>')

    input_name = sys.argv[1]
    output_name = sys.argv[2]
    col_name = sys.argv[3]
    order_char = sys.argv[4]

    # Reading data from input file
    df = pd.read_hdf(input_name, mode='r', key='features')
    with h5py.File(input_name, mode='r') as file:
        images = file['images'][:]
        masks = file['masks'][:] if 'masks' in file.keys() else None
        channels = file['channels'][:] if 'channels' in file.keys() else None

    # Sorting dataframe and images
    df, images, masks = paired_sort(df, images, masks, col_name, order_char)
    
    # Writing data to output file
    with h5py.File(output_name, mode='w') as file:
        file['images'] = images
        if channels is not None:
            file['channels'] = channels
        if masks is not None:
            file['masks'] = masks
    
    df.to_hdf(output_name, mode='a', key='features')


if __name__ == '__main__':
    
    main()
