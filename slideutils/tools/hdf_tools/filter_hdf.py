import numpy as np
import argparse
import pandas as pd
import h5py


def filter_hdf(args):
    # Read in the data
    input_file = args.input
    output_file = args.output
    filters = args.filter

    with h5py.File(input_file, "r") as f:
        images = f["images"][:]
        channels = f["channels"][:]
        masks = f["masks"][:] if "masks" in f.keys() else None

    features = pd.read_hdf(input_file, mode="r", key="features")

    # Filter the events
    n = len(features)
    sel = pd.DataFrame({"index": [True for i in range(n)]})

    for filter in filters:
        f_name = filter[0]
        f_min = float(filter[1])
        f_max = float(filter[2])
        if f_name not in features.columns:
            print(f"Cannot filter on {f_name}: Feature not found!")
            continue
        else:
            sel["index"] = (
                sel["index"]
                & (features[f_name] >= f_min)
                & (features[f_name] < f_max)
            )
            if sum(sel["index"].astype(int)) == 0:
                quit("Nothing remained after filtering!")

    features = features[sel["index"]]
    images = images[sel["index"]]
    masks = masks[sel["index"]] if masks is not None else None

    # Write the output
    with h5py.File(output_file, "w") as f:
        f.create_dataset("images", data=images)
        f.create_dataset("channels", data=channels)
        if masks is not None:
            f.create_dataset("masks", data=masks)

    features.reset_index(drop=True, inplace=True)
    features.to_hdf(output_file, key="features", mode="a")

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Filter HDF5 files",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input HDF5 files"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output HDF5 file"
    )
    parser.add_argument(
        "--filter",
        type=str,
        nargs=3,
        action="append",
        default=[],
        help="""
        feature range for filtering detected events.

        Usage:      <command> --feature_range <feature> <min> <max>
        Example:    <command> --feature_range DAPI_mean 0 10000

        Acceptable thresholds are listed in the following table:

        feature         minimum     maximum
        -------         -------     -------
        area            0           +inf
        eccentricity    0           1
        <channel>_mean  0           <MAX_VAL>
        """
    )

    args = parser.parse_args()
    filter_hdf(args)


if __name__ == "__main__":
    main()
