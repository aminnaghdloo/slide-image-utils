import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import argparse


def get_closest_events(args):
    "Find closest events to query data from reference data"

    # 1) Read inputs: reference, query, output, matching_columns
    reference = pd.read_table(args.reference, sep="\t")
    query = pd.read_table(args.query, sep="\t")

    # 2) Match both data sets on matching columns
    reference = pd.merge(query[args.matching_columns], reference, how="left")

    # 3) Find closest events to query data
    output = []
    for group_index, query_subset in query.groupby(args.matching_columns):
        ref_index = np.bitwise_and.reduce(
            reference[args.matching_columns] == group_index, axis=1
        )
        ref_subset = reference[ref_index]
        distances = cdist(query_subset[["x", "y"]], ref_subset[["x", "y"]])
        selected_index = np.argmin(distances, axis=1)
        output.append(ref_subset.iloc[selected_index, :])

    # 4) Write them in output
    output = pd.concat(output, axis=0)
    output.to_csv(args.output, index=False, sep="\t")


def main():
    parser = argparse.ArgumentParser(
        description="""
			Collect events from hdf5 that are closest to events in a 
			query dataset based on euclidean distance of event x,y coordinates.
			"""
    )

    parser.add_argument(
        "-r",
        "--reference",
        type=str,
        required=True,
        help="Path to input tab-delimited text data of reference events",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        required=True,
        help="Path to input tab-delimited text data of query events",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Path to output data"
    )
    parser.add_argument(
        "--matching_columns",
        type=str,
        nargs="*",
        default=["frame_id"],
        help="""List of column names on which events match across reference 
				and query data""",
    )

    args = parser.parse_args()
    get_closest_events(args)


if __name__ == "__main__":
    main()
