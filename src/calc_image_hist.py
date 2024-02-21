from classes import Frame
from functools import partial, update_wrapper
from skimage import measure
import multiprocessing as mp
import pandas as pd
import numpy as np
import argparse
import utils
import sys
import os


def process_frame(frame_info, params):
    "Calculates event histograms per frame to help with parallelization."
    logger = utils.get_logger("process frame", params['verbosity'])

    # function to extract events histogram
    calc_hist = utils.wrapped_partial(
        utils.calc_image_hist,

    logger.info(f"Processing frame {frame_id}...")
    frame.readImage()
    hist = utils.calc_image_hist(
        image=frame.image,
        bins=params['bins'], range=tuple(params['range']),
        density=params['density']
    )
 
    logger.info(f"Finished processing frame {frame_id}")

    return(features)


def main(args):
    
    # inputs
    image_dir   = args.image
    output      = args.output
    name_format = args.format
    starts      = args.starts
    offset      = args.offset
    n_frames    = args.nframes
    channels    = args.channels
    n_threads   = args.threads
    verbosity   = args.verbose
    include_edge= args.include_edge_frames

    # parameters for parallel frame processing
    params = {
        'channels'      : channels,
        'bins'          : args.bins,
        'range'         : args.range,
        'density'       : args.density,
        'verbosity'     : verbosity
    }

    logger = utils.get_logger(__name__, verbosity)

    logger.info("Generating frame image paths...")

    frames = []
    for i in range(n_frames):
        frame_id = i + offset + 1
        if not include_edge and utils.is_edge(frame_id):
            continue
        paths = utils.generate_tile_paths(
            path=image_dir, frame_id=frame_id, starts=starts,
            name_format=name_format)
        frame = Frame(frame_id=frame_id, channels=channels, paths=paths)
        frames.append(frame)

    logger.info("Finished generating frame image paths.")
    
    logger.info("Extracting image histograms from frames...")

    n_proc = n_threads if n_threads > 0 else mp.cpu_count()
    pool = mp.Pool(n_proc)
    hists = pool.map(partial(process_frame, params=params), frames)
    
    logger.info("Finished extracting image histograms from frames.")
    logger.info("Combining all image histograms...")

    final_hist = np.zeros(hists[0].shape, dtype='uint64')
    for hist in hists:
        final_hist += hist
    
    logger.info("Finished Combining all image histograms.")
    
    logger.info("Saving histograms...")
    df = pd.DataFrame(final_hist.transpose())
    df.rename(columns=dict(zip(df.columns, channels), inplace=True)
    df.to_csv(output, sep='\t', index=False)
    logger.info("Finished saving histograms.")

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Extract event images from coordinate data",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '-i', '--image', type=str, required=True,
        help="path to slide images directory")

    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help="path and prefix to output tab-delimited histogram file")

    parser.add_argument(
        '-r', '--range', nargs=2, type=int, default=[0, 65535],
        help="histogram range | format: min max | example: 0 65535")

    parser.add_argument(
        '-b', '--bins', type=int, default=4096,
        help="number of bins in the histogram")

    parser.add_argument(
        '-d', '--density', default=False, action='store_true',
        help="convert histograms to probability distribution function")

    parser.add_argument(
        '-f', '--offset', type=int, default=0,
        help = "start frame offset")
    
    parser.add_argument(
        '-n', '--nframes', type=int, default=2304,
        help="number of frames to process")

    parser.add_argument(
        '-c', '--channels', type=str, nargs='+',
        default=['DAPI', 'TRITC', 'CY5', 'FITC'], help="channel names")
    
    parser.add_argument(
        '-s', '--starts', type=int, nargs='+',
        default=[1, 2305, 4609, 9217], help="channel start indices")
    
    parser.add_argument(
        '-F', '--format', type=str, default='Tile%06d.tif',
        help="image name format")

    parser.add_argument(
        '-t', '--threads', type=int, default=0,
        help="number of threads for parallel processing")

    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help="verbosity level")

    parser.add_argument(
        '--include_edge_frames', default=False, action = 'store_true',
        help="include frames that are on the edge of slide")

    parser.add_argument(
        '--filter', type=str, nargs=3, action='append',
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
    logger = utils.get_logger("parse_args", args.verbose)

    # checking for potential errors in input arguments
    if not os.path.exists(args.image):
        logger.error(f"image directory {args.image} not found!")
        sys.exit(-1)

    if len(args.channels) != len(args.starts):
        logger.error(f"number of channels do not match with number of starts")
        sys.exit(-1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    logger.info("Program started")
    logger.info(f"image directory:  {args.image}")
    logger.info(f"output file:      {args.output}")

    main(args)

    logger.info("Program finished successfully!")
