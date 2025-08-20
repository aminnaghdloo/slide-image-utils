from skimage import color
from skimage import segmentation
from slideutils.utils import utils
import pandas as pd
import numpy as np
import cv2
import h5py
import os
import sys
import argparse


def create_montages(args):

    # inputs
    input = args.input
    data_path = args.data
    output = args.output
    width = args.width
    red = args.red
    green = args.green
    blue = args.blue
    order = args.order
    filters = args.filter
    sorts = args.sort
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
            # images = (images // 255).astype('uint8')
            channels = [item.decode() for item in file["channels"]]
            masks = file["masks"][:] if "masks" in file.keys() else None

            if len(images.shape) == 3:
                images = images[np.newaxis, ...]

                if masks is not None:
                    masks = masks[np.newaxis, ...]

    # checking the consistency of input arguments with input data
    if masks is None and mask_flag is not None:
        logger.error("input file does not contain 'masks'")
        sys.exit(-1)

    for channel in list(set(blue) | set(green) | set(red) | set(order)):
        if channel not in channels:
            logger.error("input channels and image channels do not match")
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

    # applying mask on images
    if mask_flag == "crop":
        images = np.multiply(images, masks)

    # cropping images to smaller size if required
    if width < images.shape[1]:
        gap = (images.shape[1] - width) // 2
        images = images[:, gap : (width + gap), gap : (width + gap), :]
        masks = masks[:, gap : (width + gap), gap : (width + gap), :] if masks is not None else None

    # apply gains
    images = utils.apply_gain(images, args.gain)

    # create montages
    logger.info("Creating the montages...")
    r_index = [channels.index(channel) for channel in red]
    g_index = [channels.index(channel) for channel in green]
    b_index = [channels.index(channel) for channel in blue]
    order_index = [channels.index(channel) for channel in order]

    if args.boundary:
        images[:, :1, :, :] = np.iinfo(images.dtype).max
        images[:, :, :1, :] = np.iinfo(images.dtype).max
        images[:, -1:, :, :] = np.iinfo(images.dtype).max
        images[:, :, -1:, :] = np.iinfo(images.dtype).max

    montages = utils.channels2montage(
        images, b_index, g_index, r_index, order_index
    )

    if mask_flag == "overlay":
        for i in range(len(montages)):
            ext_mask = np.tile(masks[i, :, :, 0], reps=1+len(order))
            marked_image = segmentation.mark_boundaries(
                montages[i], ext_mask, mode='inner', color=[0,1,1]
            )
            montages[i] = (marked_image * 255).astype('uint8')

    if args.separate:
        output_path = (
            f"{os.path.dirname(output)}/"
            f"{os.path.basename(output).split('.')[0]}"
        )
        os.makedirs(output_path, exist_ok=True)
        for i, row in df.iterrows():
            temp_path = (
                f"{output_path}/"
                f"{int(row.frame_id)}-{int(row.x)}-{int(row.y)}.jpg"
            )
            cv2.imwrite(temp_path, montages[i])
            logger.info(f"Created {temp_path}")
        df.to_csv(
            output.replace(".tif", "_montages.txt"), sep="\t", index=False
        )
    else:
        cv2.imwritemulti(output, montages)

    logger.info("Finished creating the montages!")


def main():
    parser = argparse.ArgumentParser(
        description="Create montages of events from image crops.",
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
        "-m",
        "--mask_flag",
        type=str,
        default=None,
        choices=["crop", "overlay"],
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
        "-g",
        "--gain",
        nargs="+",
        type=float,
        default=[1, 1, 1, 1],
        help="gains applied to each channel",
    )

    parser.add_argument(
        "-O",
        "--order",
        nargs="*",
        default=["DAPI", "TRITC", "FITC", "CY5"],
        help="order of channels in grayscale section of the montage",
    )

    parser.add_argument(
        "-s",
        "--separate",
        action="store_true",
        default=False,
        help="""
		save montages individually for each event as 
		<cell_id>-<frame_id>-<x>-<y>.jpg""",
    )

    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="verbosity level"
    )

    parser.add_argument(
        "-b",
        "--boundary",
        action="store_true",
        default=False,
        help="""draw white boundary around each tile in montage""",
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
    create_montages(args)

    logger.info("Program finished successfully!")


if __name__ == "__main__":
    main()
