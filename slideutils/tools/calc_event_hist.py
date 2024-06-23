from slideutils.utils.frame import Frame
from slideutils.utils import utils
from functools import partial
import multiprocessing as mp
import pandas as pd
import argparse
import sys
import os


def process_frame(frame_info, params):
    "Calculates event histograms per frame to help with parallelization."
    logger = utils.get_logger("process frame", params["verbosity"])

    # function to extract events histogram
    calc_hist = utils.wrapped_partial(
        utils.calc_event_hist,
        bins=params["bins"],
        range=tuple(params["range"]),
        density=params["density"],
    )

    (frame_id, paths) = frame_info

    logger.info(f"Processing frame {frame_id}...")
    frame = Frame(frame_id=frame_id, channels=params["channels"])
    frame.readImage(paths=paths)
    frame.readMask(
        mask_dir=params["mask_dir"], name_format=params["name_format"]
    )
    features = frame.calc_event_features(
        func=calc_hist,
        channel=params["target_channel"],
        prefix=params["prefix"],
    )
    logger.info(f"Finished processing frame {frame_id}")

    return features


def process_frames(args):

    # inputs
    image_dir = args.image
    output = args.output
    name_format = args.format
    starts = args.starts
    offset = args.offset
    n_frames = args.nframes
    channels = args.channels
    n_threads = args.threads
    filters = args.filter
    verbosity = args.verbose
    include_edge = args.include_edge_frames

    # parameters for parallel frame processing
    params = {
        "mask_dir": args.mask,
        "channels": channels,
        "target_channel": args.target_channel,
        "bins": args.bins,
        "range": args.range,
        "density": args.density,
        "prefix": args.prefix,
        "name_format": name_format,
        "verbosity": verbosity,
    }

    logger = utils.get_logger(__name__, verbosity)

    logger.info("Generating frame image paths...")
    frames_info = []
    for i in range(n_frames):
        frame_id = i + offset + 1
        if not include_edge and utils.is_edge(frame_id):
            continue
        paths = utils.generate_tile_paths(
            path=image_dir,
            frame_id=frame_id,
            starts=starts,
            name_format=name_format,
        )
        frame_info = (frame_id, paths)
        frames_info.append(frame_info)
    logger.info("Finished generating frame image paths.")

    logger.info("Extracting event histograms from frames...")

    n_proc = n_threads if n_threads > 0 else mp.cpu_count()
    pool = mp.Pool(n_proc)
    features = pool.map(partial(process_frame, params=params), frames_info)
    logger.info("Finished extracting event histograms from frames.")

    all_features = pd.concat(features, ignore_index=True)

    if len(filters) != 0:
        logger.info("Filtering events...")
        all_features = utils.filter_events(all_features, filters, verbosity)
        logger.info("Finished filtering events.")

    logger.info("Saving features...")
    all_features.to_csv(output, sep="\t", index=False)
    logger.info("Finished saving features.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract event images from coordinate data",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--image",
        type=str,
        required=True,
        help="path to slide images directory",
    )

    parser.add_argument(
        "-m", "--mask", type=str, required=True, help="path to mask directory"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="path and prefix to output tab-delimited histogram file",
    )

    parser.add_argument(
        "-r",
        "--range",
        nargs=2,
        type=int,
        default=[0, 65535],
        help="histogram range | format: min max | example: 0 65535",
    )

    parser.add_argument(
        "-b",
        "--bins",
        type=int,
        default=4096,
        help="number of bins in the histogram",
    )

    parser.add_argument(
        "-d",
        "--density",
        default=False,
        action="store_true",
        help="convert histograms to probability distribution function",
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
        default="CY5",
        help="target channel name",
    )

    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="bin",
        help="feature column name prefix",
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
        "--filter",
        type=str,
        nargs=3,
        action="append",
        default=[],
        help="""
		feature range for filtering detected events.

		Usage:		<command> --feature_range <feature> <min> <max>
		Example:	<command> --feature_range DAPI_mean 0 10000

		Acceptable thresholds are listed in the following table:

		feature		 minimum	maximum
		-------		 -------	-------
		area			0		+inf
		eccentricity	0		1
		<channel>_mean	0		<MAX_VAL>
		""",
    )

    args = parser.parse_args()
    logger = utils.get_logger("parse_args", args.verbose)

    # checking for potential errors in input arguments
    if not os.path.exists(args.image):
        logger.error(f"image directory {args.image} not found!")
        sys.exit(-1)

    if len(args.channels) != len(args.starts):
        logger.error("number of channels do not match with number of starts")
        sys.exit(-1)

    if args.target_channel not in args.channels:
        logger.error(
            f"target channel {args.target_channel} is not found in "
            f"input channels: {args.channels}"
        )
        sys.exit(-1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    args.output = f"{args.output}{args.target_channel}_hist.txt"

    logger.info("Program started")
    logger.info(f"image directory:	{args.image}")
    logger.info(f"mask directory:	{args.mask}")
    logger.info(f"output file:		{args.output}")

    process_frames(args)

    logger.info("Program finished successfully!")


if __name__ == "__main__":
    main()
