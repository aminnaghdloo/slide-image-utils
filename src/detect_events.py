#!/home/amin/miniconda3/envs/nn/bin/python
import os
import sys
import cv2
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from skimage import (
    feature, filters, measure, segmentation
)

import utils
from classes import Frame


def generate_tile_paths(path, frame_id, starts, name_format):
    paths = [f"{path}/{name_format}" % (frame_id + j - 1) for j in starts]
    return(paths)


def segment_frame(frame, params):
    "segment frame to identify all events"
    
    logger = utils.get_logger(__name__, params['verbose'])
    logger.info(f"Processing frame {frame.frame_id}...")
    
    # Preparing input
    image_copy = frame.image.copy()
    image_copy = image_copy.astype('float32')
    
    # Preparing segmentation parameters
    tophat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (params['tophat_size'], params['tophat_size']))
    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (params['opening_size'], params['opening_size']))
    
    # Preprocessing and segmenting channels separately
    target_mask = np.zeros(image_copy.shape[:2], dtype=image_copy.dtype)
    for ch in frame.channels:
        i = frame.get_ch(ch)

        image_copy[..., i] = cv2.morphologyEx(
            image_copy[..., i],
            cv2.MORPH_TOPHAT,
            tophat_kernel
        )
        image_copy[..., i] = cv2.GaussianBlur(
            image_copy[..., i],
            (params['blur_size'], params['blur_size']),
            params['blur_sigma']
        )
        
        thresh_image = filters.threshold_local(
            image=image_copy[..., i],
            method='mean',
            block_size=params['thresh_size'],
            offset=params['thresh_offset'][i]
        )
        image_copy[..., i] = image_copy[..., i] > thresh_image
        image_copy[..., i] = utils.fill_holes(
            image_copy[..., i].astype('uint8'))
        target_mask = cv2.bitwise_or(target_mask, image_copy[..., i])
    
    # Postprocessing the masks
    target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_OPEN, opening_kernel)
    target_dist = cv2.distanceTransform(
        target_mask.astype('uint8'), cv2.DIST_L2,3, cv2.CV_32F)
    
    # Generating the seeds
    seed_mask = cv2.morphologyEx(
        image_copy[..., frame.get_ch(params['seed_ch'])],
        cv2.MORPH_OPEN, opening_kernel
    )
    seed_dist = cv2.distanceTransform(
        seed_mask.astype('uint8'), cv2.DIST_L2, 3, cv2.CV_32F)
    local_max_coords = feature.peak_local_max(
        seed_dist, min_distance=params['min_dist'],
        exclude_border=params['exclude_border'])
    seeds = np.zeros(seed_mask.shape, dtype=bool)
    seeds[tuple(local_max_coords.T)] = True
    seeds = measure.label(seeds)

    # Watershed segmentation
    event_mask = segmentation.watershed(-target_dist, seeds, mask=target_mask)
    frame.mask = event_mask.astype('uint16')
    if params['mask_path'] is not None:
        frame.writeMask(params['mask_path'])
    features = utils.calc_basic_features(frame)
    
    logger.info(f"Finished processing frame {frame.frame_id}")
    
    return(features)
    

def main(args):
    
    # create logger object
    logger = utils.get_logger(__name__, args.verbose)
    
    # input variables
    in_path = args.input
    out_path = args.output
    n_frames = args.nframes
    channels = args.channels
    starts = args.starts
    offset = args.offset
    name_format = args.format
    n_threads = args.threads
    
    # segmentation parameters
    params = {
        'tophat_size': args.tophat_size,
        'opening_size': args.open_size,
        'blur_size': args.blur_size,
        'blur_sigma': args.blur_sigma,
        'thresh_size': args.thresh_size,
        'thresh_offset': args.thresh_offsets,
        'min_dist': args.min_seed_dist,
        'seed_ch': args.seed_channel,
        'mask_path': args.mask_path,
        'name_format': args.format,
        'exclude_border': args.exclude_border,
        'verbose': args.verbose 
    }
    
    # check if channel names and channel indices have same length
    if len(args.channels) != len(args.starts):
        logger.error("number of channels and number of starts do not match!")
        sys.exit(1)
    
    logger.info("Loading frame images...")
    frames = []
    for i in range(n_frames):
        frame_id = i + offset + 1
        paths = generate_tile_paths(
            path=in_path, frame_id=frame_id, starts=starts,
            name_format=name_format)
        frame = Frame(frame_id=frame_id, channels=channels)
        if frame.is_edge():
            continue
        frame.readImage(paths=paths)
        frames.append(frame)
    logger.info("Finished loading frame images.")
    
    logger.info("Processing the frames...")
    n_proc = n_threads if n_threads > 0 else mp.cpu_count()
    pool = mp.Pool(n_proc)
    features = pool.map(partial(segment_frame, params=params), frames)
    logger.info("Finished processing the frames.")
    
    logger.info("Saving features...")
    feature_vec = pd.concat(features)
    feature_vec.to_csv(out_path, sep='\t')
    logger.info("Finished saving features.")

if __name__ == '__main__':
    
    # main inputs
    parser = argparse.ArgumentParser(
        description="Process slide images to identify cells.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input', type=str, required=True,
        help="input path to slide images")
    parser.add_argument(
        '-o', '--output', type=str, required=True, help="output path")
    parser.add_argument(
        '-m', '--mask_path', type=str, default=None,
        help="mask path to save frame masks if needed")
    parser.add_argument(
        '-f', '--offset', type=int, default=0, help="start frame offset")
    parser.add_argument(
        '-n', '--nframes', type=int, default=2304, help="number of frames")
    parser.add_argument(
        '-c', '--channels', type=str, nargs='+',
        default=['DAPI', 'TRITC', 'Cy5', 'FITC'], help="channel names")
    parser.add_argument(
        '-s', '--starts', type=int, nargs='+',
        default=[1, 2305, 4609, 9217], help="channel start indices")
    parser.add_argument(
        '-F', '--format', type=str, default='Tile%06d.tif',
        help="image name format")
    parser.add_argument(
        '-v', '--verbose', action='count', default=0, help="verbosity level")
    parser.add_argument(
        '-t', '--threads', type=int, default=0,
        help="number of threads for parallel processing")
    
    # Segmentation parameters
    parser.add_argument(
        '--tophat_size', type=int, default=45,
        help="TopHat filter kernel size")
    parser.add_argument(
        '--open_size', type=int, default=5,
        help="Open morphological filter kernel size")
    parser.add_argument(
        '--blur_size', type=int, default=5,
        help="Gaussian blur filter kernel size")
    parser.add_argument(
        '--blur_sigma', type=int, default=2,
        help="Gaussian blur filter parameter sigma")
    parser.add_argument(
        '--thresh_size', type=int, default=25,
        help="adaptive threshold window size")
    parser.add_argument(
        '--thresh_offsets', type=int, nargs='+',
        default=[-2000, -2000, -2000, -2000],
        help="adaptive threshold offset values for channels (negative)")
    parser.add_argument(
        '--min_seed_dist', type=int, default=7,
        help="minimum allowed distance [pixels] between each pair of seeds")
    parser.add_argument(
        '--seed_channel', type=str, default='DAPI',
        help="channel name to use as seed")
    parser.add_argument(
        '--size_filter_min', type=int, default=0,
        help="minimum size [in pixels] to filter events")
    parser.add_argument(
        '--exclude_border', default=False, action='store_true',
        help="exclude events that are on image borders")
    

    args = parser.parse_args()

    main(args)