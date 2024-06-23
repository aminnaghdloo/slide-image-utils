from scipy.spatial import KDTree
import numpy as np
import pandas as pd
import sys


def main():
    """
    Calculate the minimum distance between each data point in query data and
    the two closest data points in reference data. Both query and reference
    data files require columns x and y.
    """

    # Load inputs
    if len(sys.argv) != 3:
        print("Usage: python calc_min_dist.py <query.txt> <reference path>")
        sys.exit(-1)

    query_file_name = sys.argv[1]
    ref_dir_name = sys.argv[2]

    data_types = {"slide_id": str, "frame_id": int, "x": float, "y": float}
    df_q = pd.read_table(query_file_name, sep="\t", dtype=data_types)

    dist_dict = {
        "r": [],
        "cell_idw": [],
        "xw": [],
        "yw": [],
        "rw": [],
        "min_dist": [],
        "gap": [],
        "cell_idw2": [],
        "xw2": [],
        "yw2": [],
        "rw2": [],
        "min_dist2": [],
        "gap2": [],
        "gap_c": [],
        "min_dist_c": [],
        "dapi_count": [],
    }

    for index, row in df_q.iterrows():

        # read the reference file
        df_r = pd.read_table(
            f"{ref_dir_name}/{row['slide_id']}_{row['frame_id']}.txt"
        )
        dapi_count = len(df_r)

        # get the coordinates of the query and reference data points
        ref_coords = list(zip(df_r["x"], df_r["y"]))
        query_coord = (row["x"], row["y"])

        # get the two closest WBCs
        tree = KDTree(ref_coords)
        distances, indices = tree.query(query_coord, k=2)

        # save the distances and coordinates
        r = (row["area"] / np.pi) ** 0.5
        rw = (df_r.iloc[indices[0]]["area"] / np.pi) ** 0.5
        rw2 = (df_r.iloc[indices[1]]["area"] / np.pi) ** 0.5
        gap = distances[0] - r - rw
        gap2 = distances[1] - r - rw2

        if -gap > r + rw - abs(r - rw) and -gap < r + rw:
            gap_c = gap2
            min_dist_c = distances[1]
        else:
            gap_c = gap
            min_dist_c = distances[0]

        dist_dict["r"].append(r)
        dist_dict["cell_idw"].append(df_r.iloc[indices[0]]["cell_id"])
        dist_dict["xw"].append(df_r.iloc[indices[0]]["x"])
        dist_dict["yw"].append(df_r.iloc[indices[0]]["y"])
        dist_dict["rw"].append(rw)
        dist_dict["min_dist"].append(distances[0])
        dist_dict["gap"].append(gap)
        dist_dict["cell_idw2"].append(df_r.iloc[indices[1]]["cell_id"])
        dist_dict["xw2"].append(df_r.iloc[indices[1]]["x"])
        dist_dict["yw2"].append(df_r.iloc[indices[1]]["y"])
        dist_dict["rw2"].append(rw2)
        dist_dict["min_dist2"].append(distances[1])
        dist_dict["gap2"].append(gap2)
        dist_dict["gap_c"].append(gap_c)
        dist_dict["min_dist_c"].append(min_dist_c)
        dist_dict["dapi_count"].append(dapi_count)

        print(
            f"processed {index} / {len(df_q)} against {len(df_r)} references"
        )

    dist_df = pd.DataFrame(dist_dict)
    out_df = pd.concat([df_q.iloc[:, :6], dist_df], axis=1)
    out_df.to_csv(
        f"{query_file_name[:-4]}_min_dist.txt", sep="\t", index=False
    )
    print(f"Minimum distances were calculated for {query_file_name}")


if __name__ == "__main__":
    main()
