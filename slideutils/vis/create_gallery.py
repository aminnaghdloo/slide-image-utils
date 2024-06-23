from skimage import color
import pandas as pd
import numpy as np
import cv2
import math
import h5py
import os
import sys
import argparse
from slideutils.utils import utils


def create_gallery(images, n_x, n_y):
    "Create 2D gallery from list of images."
    if len(images.shape) == 3:
        images = images[np.newaxis, ...]
    shape = images.shape
    h = shape[1] * n_y
    w = shape[2] * n_x
    d = shape[3]
    pages = math.ceil(len(images) / (n_y * n_x))
    n_total = pages * n_y * n_x
    n_current = len(images)
    if n_total != n_current:
        filler = np.zeros(
            (n_total - n_current, shape[1], shape[2], d), dtype=images.dtype
        )
        images = np.append(images, filler, axis=0)
    gallery = np.zeros((pages, h, w, d), dtype="uint16")
    for k in range(pages):
        gallery[k, :, :, :] = np.vstack(
            [
                np.hstack(
                    [
                        images[(k * n_x * n_y) + (j * n_x) + i, :, :, :]
                        for i in range(n_x)
                    ]
                )
                for j in range(n_y)
            ]
        )
    return gallery


def main_process(args):

    # inputs
    input = args.input
    data_path = args.data
    output = args.output
    width = args.width
    n_x = args.nx
    n_y = args.ny
    red = args.red
    green = args.green
    blue = args.blue
    sorts = args.sort
    filters = args.filter
    verbosity = args.verbose
    mask_flag = args.mask_flag

    logger = utils.get_logger(__name__, verbosity)

    # reading input data
    if data_path is None:
        df = pd.read_hdf(input, key="features")
    else:
        df = pd.read_table(data_path, sep="\t")

    with h5py.File(input, "r") as file:
        if "images" not in file.keys():
            logger.error("input file does not contain 'images'")
            sys.exit(-1)
        elif "channels" not in file.keys():
            logger.error("input file does not contain 'channels'")
            sys.exit(-1)
        else:
            images = file["images"][:]
            channels = [item.decode() for item in file["channels"]]
            masks = file["masks"][:] if "masks" in file.keys() else None

    # checking the consistency of input arguments with input data
    if masks is None and mask_flag != 0:
        logger.error("input file does not contain 'masks'")
        sys.exit(-1)

    for channel in list(set(blue) | set(green) | set(red)):
        if channel not in channels:
            logger.error(
                "input color channels and image channels do not match"
            )
            sys.exit(-1)

    # applying the input filters
    if len(filters) != 0:
        logger.info("Filtering events...")
        df = utils.filter_events(df, filters, verbosity)
        logger.info("Finished filtering events.")
    images = images[list(df.index)]
    masks = masks[list(df.index)] if masks is not None else None
    df.reset_index(drop=True, inplace=True)

    # applying the input sortings
    if len(sorts) != 0:
        logger.info("Sorting events...")
        utils.sort_events(df, sorts, verbosity)
        logger.info("Finished sorting events.")
    images = images[list(df.index)]
    masks = masks[list(df.index)] if masks is not None else None
    df.reset_index(drop=True, inplace=True)

    # converting to BGR image for visualization
    logger.info("Creating the gallery...")
    red_index = [channels.index(channel) for channel in red]
    green_index = [channels.index(channel) for channel in green]
    blue_index = [channels.index(channel) for channel in blue]
    images = utils.channels_to_bgr(images, blue_index, green_index, red_index)

    # applying mask on images
    if mask_flag == 1:
        images = np.multiply(images, (masks != 0).astype(int))
    elif mask_flag == 2:
        images = color.label2rgb(
            label=masks[..., 0], image=images, channel_axis=3
        )
        images = (images * 65535).astype("uint16")

    # cropping images to smaller size if required
    if width < images.shape[1]:
        gap = (images.shape[1] - width) // 2
        images = images[:, gap : (width + gap), gap : (width + gap), :]

    # create gallery
    gallery = create_gallery(images=images, n_x=n_x, n_y=n_y)
    gallery = (gallery // 256).astype("uint8")

    cv2.imwritemulti(output, gallery)
    logger.info("Finished creating the gallery!")


def main():
    parser = argparse.ArgumentParser(
        description="Create gallery of events from image crops.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="path to HDF file containing event images",
    )

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=None,
        help="path to tab-delimited events data file",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="path to output gallery image",
    )

    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=35,
        help="size of each event image crop (odd)",
    )

    parser.add_argument(
        "-x",
        "--nx",
        type=int,
        default=15,
        help="number of images along x axis in gallery",
    )

    parser.add_argument(
        "-y",
        "--ny",
        type=int,
        default=15,
        help="number of images along y axis in gallery",
    )

    parser.add_argument(
        "-m",
        "--mask_flag",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="mask flag",
    )

    parser.add_argument(
        "-R",
        "--red",
        nargs="+",
        required=True,
        default=[],
        help="channel(s) to be shown in red color",
    )

    parser.add_argument(
        "-G",
        "--green",
        nargs="+",
        required=True,
        default=[],
        help="channel(s) to be shown in green color",
    )

    parser.add_argument(
        "-B",
        "--blue",
        nargs="+",
        required=True,
        default=[],
        help="channel(s) to be shown in blue color",
    )

    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="verbosity level"
    )

    parser.add_argument(
        "--sort",
        type=str,
        nargs=2,
        action="append",
        default=[],
        help="""
		sort events based on feature values.

		Usage:	  <command> --sort <feature> <order>
		Example:	<command> --sort TRITC_mean I
		order:	  I: Increasing / D: Decreasing
		""",
    )

    parser.add_argument(
        "--filter",
        type=str,
        nargs=3,
        action="append",
        default=[],
        help="""
		feature range for filtering detected events.

		Usage:	  <command> --feature_range <feature> <min> <max>
		Example:	<command> --feature_range DAPI_mean 0 10000

		Acceptable thresholds are listed in the following table:

		feature		 minimum	 maximum
		-------		 -------	 -------
		area			0		   +inf
		eccentricity	0		   1
		<channel>_mean  0		   <MAX_VAL>
		""",
    )

    args = parser.parse_args()
    logger = utils.get_logger("parse_args", args.verbose)

    # checking for potential errors in input arguments
    if not os.path.exists(args.input):
        logger.error(f"image data file {args.input} not found!")
        sys.exit(-1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    logger.info("Program started")
    logger.info(f"input image file: {args.input}")
    logger.info(f"input data file:  {args.data}")
    logger.info(f"output file:	  {args.output}")

    main_process(args)

    logger.info("Program finished successfully!")


if __name__ == "__main__":
    main()
