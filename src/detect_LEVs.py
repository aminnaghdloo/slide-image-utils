from skimage import segmentation, measure
from functools import partial
from classes import Frame
import pandas as pd
import numpy as np
import multiprocessing as mp
import cv2
import h5py
import argparse
import sys
import os
import utils


def process_frame(frame_info, params):
    "Process frame to identify target LEVs"
     
    logger = utils.get_logger("process_frame", params['verbosity'])
    (frame_id, paths) = frame_info
    logger.info(f"Processing frame {frame_id}...")

    # loading frame image
    frame = Frame(frame_id=frame_id, channels=params['channels'])
    frame.readImage(paths=paths)
    image_copy = frame.image.copy()
    image_copy = image_copy.astype('float32')

    # preprocessing image
    tophat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (params['tophat_size'], params['tophat_size']))

    for i in range(len(frame.channels)):
        image_copy[..., i] = cv2.morphologyEx(
            image_copy[..., i],
            cv2.MORPH_TOPHAT,
            tophat_kernel
        )

    # image segmentation using double thresholding
    target_image = image_copy[..., params['channel_id']]
    th1 = np.percentile(target_image, params['low_thresh'])
    ret, foreground = cv2.threshold(
        target_image, th1, params['max_val'], cv2.THRESH_BINARY)
    masked_image = target_image[foreground != 0]
    th2 = params['high_thresh'] * np.median(masked_image)
    ret, seeds = cv2.threshold(
        target_image, th2, params['max_val'], cv2.THRESH_BINARY)
    seeds = measure.label(seeds)
    mask = segmentation.watershed(-foreground, seeds, mask=foreground)
    frame.mask = mask.astype('uint16')

    # storing mask
    if params['mask_dir'] is not None:
        frame.writeMask(params['mask_dir'], name_format=params['name_format'])
    
    # extracting features
    features = utils.calc_basic_features(frame)
    logger.info(f"Finished processing frame {frame.frame_id}")

    return(features)


def main(args):
    
    # inputs
    input       = args.input
    output      = args.output
    starts      = args.starts
    offset      = args.offset
    n_frames    = args.nframes
    n_threads   = args.threads
    name_format = args.format
    filters     = args.filter
    verbosity   = args.verbose

    logger = utils.get_logger(__name__, verbosity)

    
    # parameters for process_frame function
    params = {
        'channels': args.channels,
        'tophat_size': args.kernel,
        'channel_id': args.channels.index(args.target_channel),
        'max_val': args.max_val,
        'low_thresh': args.low,
        'high_thresh': args.high,
        'mask_dir': args.mask,
        'name_format': args.format,
        'verbosity': verbosity
    }

    logger.info("Generating frame image paths...")
    frames_info = []
    for i in range(n_frames):
        frame_id = i + offset + 1
        if utils.is_edge(frame_id):
            continue
        paths = utils.generate_tile_paths(
            path=input, frame_id=frame_id, starts=starts,
            name_format=name_format)
        frame_info = (frame_id, paths)
        frames_info.append(frame_info)
    logger.info("Finished generating frame image paths.")

    logger.info("Processing the frames...")
    n_proc = n_threads if n_threads > 0 else mp.cpu_count()
    pool = mp.Pool(n_proc)
    features = pool.map(partial(process_frame, params=params), frames_info)
    logger.info("Finished processing the frames.")
    
    all_features = pd.concat(features, ignore_index=True)

    if(len(filters) != 0):
        logger.info("Filtering events...")
        all_features = utils.filter_events(all_features, filters, verbosity)
        logger.info("Finished filtering events.")
    
    logger.info("Saving features...")
    all_features.to_csv(output, sep='\t', index=False)
    logger.info(f"Finished saving features of {len(all_features)} events.")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Process slide images to detect LEVs with a single channel",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '-i', '--input', type=str, required=True,
        help="path to slide images directory")

    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help="output file path")

    parser.add_argument(
        '-m', '--mask', type=str, default=None,
        help="path to a directory to save event masks [optional]")

    parser.add_argument(
        '-f', '--offset', type=int, default=0,
        help="start frame offset")

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
        '-r', '--target_channel', type=str, default='TRITC',
        help="target channel id for LEV detection")

    parser.add_argument(
        '-L', '--low', type=float, default=99.9,
        help="low threshold for target channel segmentation [percentile]")

    parser.add_argument(
        '-H', '--high', type=float, default=2,
        help="high threshold for target channel segmentation [ratio-to-median]")

    parser.add_argument(
        '-k', '--kernel', type=int, default=45,
        help="size of tophat filter kernel")

    parser.add_argument(
        '--max_val', type=int, default=65535,
        help="maximum pixel value for foreground during thresholding")

    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help="verbosity level")

    parser.add_argument(
        '--filter', type=str, nargs=3, action='append',
        default=[],
        #default=[['area', '15', '5000'], ['DAPI_mean', '0', '10000']],
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
    if not os.path.exists(args.input):
        logger.error(f"input {args.input} not found!")
        sys.exit(-1)
    
    if args.target_channel not in args.channels:
        logger.error(f"target channel is not found among channels!")
        sys.exit(-1)

    if len(args.channels) != len(args.starts):
        logger.error(f"number of channels do not match with number of starts")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    logger.info("Program started")
    logger.info(f"input:    {args.input}")
    logger.info(f"output:   {args.output}")

    main(args)

    logger.info("Program finished successfully!")





#with h5py.File(f"{data_path}/{slide_id}/{slide_id}_LEVs.hdf5", "w") as hf:
#    hf.create_dataset('images', data=images)

#features.to_hdf(f"{data_path}/{slide_id}/{slide_id}_LEVs.hdf5", key='features')

#print(f"Successful EV extraction from {slide_id}! (total: {len(images)})")
#width
#mask_flag
