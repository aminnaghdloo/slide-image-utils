import numpy as np
import pandas as pd
import sys
import os
import glob
import cv2
import tifffile as tff

# This program collects the mongates of 40x frames of reimaged events
# input: slide_list.txt
# output: <slide_id>_<frame_id>_<event_x>_<event_y>.tif for each 40x frame


# global variables:
path_template = (
    "/mnt/Y/DZ/{}/ClientBin/40X"  # template for path reimaging data
)
ch_list = ["DAPI", "CY5", "TRITC"]


# functions
def main():

    # read command line arguments
    args = sys.argv

    # check arguments
    if len(args) != 2:
        quit(f"Usage: python query_40x_images.py slides.txt")

    # read input file to get the list of slides
    file_name = args[1]
    with open(file_name, mode="r") as file:
        slides = file.read().splitlines()

    # collect event data
    event_data = {
        "slide_id": [],
        "frame_id": [],
        "cell_id": [],
        "x": [],
        "y": [],
    }

    # process slides
    for slide in slides:

        slide_path = path_template.format(slide)

        # check if the slide is reimaged
        if not os.path.exists(slide_path):
            print(f"Path does not exist for {slide}!")
            continue

        # collect 40x frame image names from metadata files (txt files)
        image_metadata = glob.glob(f"{slide_path}/*-*-*-*.txt")
        image_names = [item.split(".txt")[0] for item in image_metadata]

        # process images
        for image_name in image_names:

            # create paths to channel images
            ch_paths = [f"{image_name}-{ch}.tif" for ch in ch_list]

            # read channel images (DAPI, CY5, TRITC)
            image = [
                tff.imread(ch_path)
                for ch_path in ch_paths
                if os.path.exists(ch_path)
            ]
            image = np.stack(image, axis=2)

            # check if all channels exist
            if image.shape[2] != 3:
                print(f"Missing channel images from {slide}...")
                continue

            # check if they are 8-bit, otherwise convert them to 8-bit
            if image.dtype == "uint16":
                image = np.right_shift(image, 4).astype("uint8")

            # collect event data from image name
            cell, frame, x, y = os.path.basename(image_name).split("-")

            # save image
            cv2.imwrite(f"{slide}_{frame}_{x}_{y}.jpg", image)

            # save frame data into a file (slide_id, frame_id, x, y)
            event_data["slide_id"].append(slide)
            event_data["frame_id"].append(frame)
            event_data["cell_id"].append(cell)
            event_data["x"].append(x)
            event_data["y"].append(y)

        print(f"Successfully queried {slide}")

    df = pd.DataFrame(event_data)
    df.to_csv(
        f"event_data_{os.path.basename(file_name)}", index=False, sep="\t"
    )


if __name__ == "__main__":
    main()
