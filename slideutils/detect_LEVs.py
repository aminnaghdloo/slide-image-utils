from skimage import measure, morphology
from scipy.ndimage import binary_fill_holes as fill_hole
from functools import partial
from slideutils.utils.frame import Frame
from slideutils.utils import utils
import pandas as pd
import numpy as np
import multiprocessing as mp
import cv2
import h5py
import argparse
import sys
import os


def process_frame(frame, params):
    "Process frame to identify target LEVs"

    logger = utils.get_logger("process_frame", params["verbosity"])
    logger.info(f"Processing frame {frame.frame_id}...")

    # loading frame image
    frame.readImage()
    image_copy = frame.image.copy()
    image_copy = image_copy.astype("float32")

    # preprocessing image
    if params["tophat_size"] != 0:
        tophat_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (params["tophat_size"], params["tophat_size"])
        )

        for i in range(len(frame.channels)):
            image_copy[..., i] = cv2.morphologyEx(
                image_copy[..., i], cv2.MORPH_TOPHAT, tophat_kernel
            )

    # image segmentation using double thresholding
    target_image = image_copy[..., params["channel_id"]]
    target_image = cv2.bilateralFilter(
        target_image.astype(np.float32), 15, 75, 75
    )
    target_image = target_image.astype("uint16")
    th1 = np.percentile(target_image, params["low_thresh"])
    ret1, foreground = cv2.threshold(
        target_image, th1, params["max_val"], cv2.THRESH_BINARY
    )
    foreground = (fill_hole(foreground) * params["max_val"]).astype("uint16")
    masked_image = target_image[foreground != 0]
    th2 = params["high_thresh"] * np.median(masked_image)
    ret2, seeds = cv2.threshold(
        target_image, th2, params["max_val"], cv2.THRESH_BINARY
    )
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, opening_kernel)
    seeds = cv2.bitwise_and(seeds, foreground)  # added
    mask = morphology.reconstruction(seeds, foreground)
    mask = measure.label(mask)
    frame.mask = mask.astype("uint16")

    # storing mask
    if params["mask_dir"] is not None:
        frame.writeMask(params["mask_dir"])

    # extracting features
    features = frame.calc_basic_features()

    if features is not None:
        for ch in frame.channels:
            temp = frame.calc_event_features(
                func=utils.calc_percentiles,
                channel=ch,
                columns=["q05", "q10", "q25", "q50", "q75", "q90", "q95"],
            )
            features = features.merge(temp, on=("frame_id", "cell_id", "y", "x"))

    images = None
    masks = None

    # filtering events
    if len(params["filters"]) != 0:
        features = utils.filter_events(
            features, params["filters"], params["verbose"]
        )

    # extracting event images
    if features is not None:
        if params["extract_img"]:
            images, masks = frame.extract_crops(
                features, params["width"], mask_flag=params["mask_flag"]
            )

    # extracting background mean intensities:
    # x = utils.calc_bg_intensity(frame)
    # x = x.reindex(x.index.repeat(len(features))).reset_index(drop=True)
    # features = pd.concat([features, x], axis=1)

    logger.info(f"Finished processing frame {frame.frame_id}")
    return {"features": features, "images": images, "masks": masks}


def process_frames(args):
    # inputs
    input = args.input
    output = args.output
    starts = args.starts
    offset = args.offset
    n_frames = args.nframes
    n_threads = args.threads
    name_format = args.format
    verbosity = args.verbose
    include_edge = args.include_edge_frames
    extract_img = args.extract_images
    width = args.width
    mask_flag = args.mask_flag
    sorts = args.sort

    logger = utils.get_logger(__name__, verbosity)

    # parameters for process_frame function
    params = {
        "tophat_size": args.kernel,
        "channel_id": args.channels.index(args.target_channel),
        "bg_cell_channels": args.bg_cell_channels,
        "max_val": args.max_val,
        "low_thresh": args.low,
        "high_thresh": args.high,
        "mask_dir": args.mask,
        "name_format": args.format,
        "filters": args.filter,
        "extract_img": extract_img,
        "mask_flag": mask_flag,
        "width": width,
        "verbosity": verbosity,
    }

    logger.info("Generating frame image paths...")
    frames = []
    for i in range(n_frames):
        frame_id = i + offset + 1
        if not include_edge and utils.is_edge(frame_id):
            continue
        paths = utils.generate_tile_paths(
            path=input,
            frame_id=frame_id,
            starts=starts,
            name_format=name_format,
        )
        frame = Frame(frame_id=frame_id, channels=args.channels, paths=paths)
        frames.append(frame)
    logger.info("Finished generating frame image paths.")

    logger.info("Processing the frames...")
    n_proc = n_threads if n_threads > 0 else mp.cpu_count()
    mp.set_start_method('fork')
    pool = mp.Pool(n_proc)
    data = pool.map(partial(process_frame, params=params), frames)
    logger.info("Finished processing the frames.")

    logger.info("Collecting features...")
    all_features = [
        out["features"] for out in data if out["features"] is not None
        ]
    if len(all_features) == 0:
        logger.error("No event to report on this slide!")
        sys.exit(-1)
    else:
        all_features = pd.concat(all_features, ignore_index=True)

    all_images = None
    all_masks = None
    if extract_img:
        logger.info("Collecting event images...")
        all_images = np.concatenate(
            [out["images"] for out in data if out["images"] is not None],
            axis=0,
        )

        if mask_flag:
            logger.info("Collecting event masks...")
            all_masks = np.concatenate(
                [out["masks"] for out in data if out["masks"] is not None],
                axis=0,
            )

    # applying the input sortings
    if len(args.sort) != 0:
        logger.info("Sorting events...")
        utils.sort_events(all_features, args.sort, args.verbose)
        logger.info("Finished sorting events.")
        all_images = all_images[list(all_features.index)]
        all_masks = all_masks[list(all_features.index)]
        all_features.reset_index(drop=True, inplace=True)

    logger.info("Saving data...")
    if not extract_img:
        all_features.round(decimals=3).to_csv(output, sep="\t", index=False)
    else:
        output = output.replace(".txt", ".hdf5")
        with h5py.File(output, "w") as hf:
            hf.create_dataset("images", data=all_images)
            hf.create_dataset("channels", data=args.channels)
            if mask_flag:
                hf.create_dataset("masks", data=all_masks)

        all_features.to_hdf(output, mode="a", key="features")

    logger.info("Finished saving data.")


def main():
    parser = argparse.ArgumentParser(
        description="Process slide images for marker-based detection of LEVs",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="path to slide images directory",
    )

    parser.add_argument(
        "-o", "--output", type=str, required=True, help="output file path"
    )

    parser.add_argument(
        "-m",
        "--mask",
        type=str,
        default=None,
        help="path to a directory to save event masks [optional]",
    )

    parser.add_argument(
        "-f", "--offset", type=int, default=0, help="start frame offset"
    )

    parser.add_argument(
        "-n",
        "--nframes",
        type=int,
        default=2304,
        help="number of frames to process",
    )

    parser.add_argument(
        "-c",
        "--channels",
        type=str,
        nargs="+",
        default=["DAPI", "TRITC", "CY5", "FITC"],
        help="channel names",
    )

    parser.add_argument(
        "-C",
        "--bg_cell_channels",
        type=str,
        nargs="+",
        default=["DAPI", "CY5"],
        help="name of channels for which background cells are positive",
    )

    parser.add_argument(
        "-s",
        "--starts",
        type=int,
        nargs="+",
        default=[1, 2305, 4609, 9217],
        help="channel start indices",
    )

    parser.add_argument(
        "-F",
        "--format",
        type=str,
        default="Tile%06d.tif",
        help="image name format",
    )

    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=0,
        help="number of threads for parallel processing",
    )

    parser.add_argument(
        "-T",
        "--target_channel",
        type=str,
        default="TRITC",
        help="target channel name for LEV detection",
    )

    parser.add_argument(
        "-L",
        "--low",
        type=float,
        default=99.7,
        help="low threshold for segmentation [percentile]",
    )

    parser.add_argument(
        "-H",
        "--high",
        type=float,
        default=2,
        help="high threshold for segmentation [ratio-to-median]",
    )

    parser.add_argument(
        "-k",
        "--kernel",
        type=int,
        default=75,
        help="size of tophat filter kernel",
    )

    parser.add_argument(
        "--max_val",
        type=int,
        default=65535,
        help="maximum pixel value for foreground during thresholding",
    )

    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="verbosity level"
    )

    parser.add_argument(
        "--include_edge_frames",
        default=False,
        action="store_true",
        help="include frames that are on the edge of slide",
    )

    parser.add_argument(
        "--extract_images",
        default=False,
        action="store_true",
        help="extract images of detected events and output hdf5 file",
    )

    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=45,
        help="""
		size of the event images to be cropped from slide images (odd).
		Works only when --extract_images is set.
		""",
    )

    parser.add_argument(
        "--mask_flag",
        default=False,
        action="store_true",
        help="store event masks when extracting images",
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
        logger.error(f"input {args.input} not found!")
        sys.exit(-1)

    if args.target_channel not in args.channels:
        logger.error("target channel is not found among channels!")
        sys.exit(-1)

    if len(args.channels) != len(args.starts):
        logger.error("number of channels do not match with number of starts")

    for item in args.bg_cell_channels:
        if item not in args.channels:
            logger.error(
                f"background cell channel {item} is not found among"
                f" image channels!"
            )
            sys.exit(-1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    logger.info("Program started")
    logger.info(f"input:	{args.input}")
    logger.info(f"output:   {args.output}")

    process_frames(args)

    logger.info("Program finished successfully!")


if __name__ == "__main__":
    main()
