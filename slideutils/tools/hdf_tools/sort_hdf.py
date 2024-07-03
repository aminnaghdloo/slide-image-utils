import sys
import h5py
import pandas as pd
import numpy as np
import argparse


def paired_sort(df, images, masks, column, order_char):
    "A function that sorts images and masks"
    df.sort_values(
        column, inplace=True, ascending=True if order_char == "A" else False
    )
    images = images[list(df.index), ...]
    masks = masks[list(df.index), ...] if masks is not None else None

    df.reset_index(inplace=True, drop=True)
    return df, images, masks


def main():
    parser = argparse.ArgumentParser(
        description="sort HDF5 files",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input HDF5 file"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output HDF5 file" 
    )
    parser.add_argument(
        "-c", "--column", type=str, required=True, help="column to sort"
    )
    parser.add_argument(
        "-o", "--order", type=str, required=True, choices=['A', 'D'],
        help="sorting order: A -> ascending / D -> descending"
    )

    args = parser.parse_args()

    # Reading data from input file
    df = pd.read_hdf(args.input, mode="r", key="features")
    with h5py.File(args.input, mode="r") as file:
        images = file["images"][:]
        masks = file["masks"][:] if "masks" in file.keys() else None
        channels = file["channels"][:] if "channels" in file.keys() else None

    # Sorting dataframe and images
    df, images, masks = paired_sort(df, images, masks, args.column, args.order)

    # Writing data to output file
    with h5py.File(args.output, mode="w") as f:
        f.create_dataset("images", data=images)
        f.create_dataset("channels", data=channels)
        if masks is not None:
            f.create_dataset("masks", data=masks)

    df.reset_index(drop=True, inplace=True)
    df.to_hdf(args.output, mode="a", key="features")
    print("Done!")


if __name__ == "__main__":
    main()
