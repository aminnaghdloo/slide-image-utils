import pandas as pd
import argparse
import sys


def hdf2txt(hdf_file, txt_file):
    """Convert hdf5 file to txt file"""
    df = pd.read_hdf(hdf_file, mode="r", key="features")
    #df = df[df.pred == 1]
    #df["slide_id"] = hdf_file.split("/")[-1].split(".")[0]
    df.to_csv(txt_file, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Extract tabular features from hdf5 files",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="path to input hdf5 file"
    )

    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="path to output tab-delimited file"
    )

    args = parser.parse_args()

    hdf2txt(args.input, args.output)
    print(
        f"Successfully extracted features from {args.input} to {args.output}!"
    )


if __name__ == "__main__":
    main()
