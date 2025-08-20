import numpy as np
import argparse
import pandas as pd
import h5py

def merge_hdf(args):
    # Read in the data
    all_images = []
    all_features = []
    all_masks = []
    all_channels = []

    for input_file in args.input:

        filename = input_file.split("/")[-1].split(".")[0]

        if args.verbose:
            print(f"Reading {input_file}...")

        with h5py.File(input_file, "r") as f:
            all_images.append(f["images"][:])
            if args.mask_flag:
                all_masks.append(f["masks"][:])
            channels = f["channels"][:]

        df = pd.read_hdf(input_file, mode="r", key="features")
        if args.add_filename_column is not None:
            df.insert(0, args.add_filename_column, filename)
        all_features.append(df)

    # Concatenate the data
    images = np.concatenate(all_images, axis=0)
    if args.mask_flag:
        masks = np.concatenate(all_masks, axis=0)
    features = pd.concat(all_features, axis=0)

    # Write out the data
    if args.verbose:
        print(f"Writing {args.output}...")

    with h5py.File(args.output, "w") as f:
        f.create_dataset("images", data=images)
        if args.mask_flag:
            f.create_dataset("masks", data=masks)
        f.create_dataset("channels", data=channels)

    features.reset_index(drop=True, inplace=True)
    features.to_hdf(args.output, key="features", mode="a")

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Merge HDF5 files")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Input HDF5 files",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output HDF5 file", required=True
    )
    parser.add_argument(
        "-m",
        "--mask_flag",
        action="store_true",
        default=False,
        help="merge the masks as well",
    )
    parser.add_argument(
        "-a",
        "--add_filename_column",
        type=str,
        default=None,
        help="column name to which the code adds file names",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose mode"
    )

    args = parser.parse_args()

    merge_hdf(args)

if __name__ == "__main__":
    main()
