from classes import Frame
import pandas as pd
import numpy as np
import argparse
import utils
import h5py
import os


def main(args):
    
    # inputs
    image_dir   = args.image
    mask_dir    = args.mask
    data_path   = args.data
    output      = args.output
    width       = args.width
    name_format = args.format
    starts      = args.starts
    channels    = args.channels
    filters     = args.filter
    verbosity   = args.verbose
    mask_flag   = False if mask_dir is None else True
    
    logger = utils.get_logger(__name__, verbosity)
    df = pd.read_table(data_path)
    df['frame_id'] = df['frame_id'].astype(int)

    if(len(filters) != 0):
        logger.info("Filtering events...")
        df = utils.filter_events(df, filters, verbosity)
        logger.info("Finished filtering events.")
    
    image_ids = list(range(len(df)))
    df.insert(0, 'image_id', image_ids)
    all_images = np.zeros((len(df), width, width, len(channels)),
                          dtype='uint16')
    if mask_flag:
        all_masks = np.zeros((len(df), width, width, len(channels)),
                             dtype='uint16')
    
    groups = df.groupby('frame_id')
    for frame_id, event_data in groups:
        frame = Frame(frame_id, channels)
        paths = utils.generate_tile_paths(
            image_dir, frame_id, starts, name_format)
        frame.readImage(paths)
        if mask_flag:
            frame.readMask(mask_dir, name_format=name_format)
        images, masks = frame.extract_crops(event_data, width, mask_flag)
        
        index = groups.indices[frame_id]
        all_images[index.tolist(), :, :, :] = images
        
        if mask_flag:
            all_masks[index.tolist(), :, :, :] = masks

        logger.info(f"{len(event_data)} images extracted from frame {frame_id}")

    logger.info('Saving extracted images...')
    with h5py.File(output, 'w') as file:
        file.create_dataset('images', data=all_images)
        file.create_dataset('channels', data=channels)
        if mask_flag:
            file.create_dataset('masks', data=all_masks)
    
    df.to_hdf(output, mode='a', key='features')
    logger.info('Finished saving extracted images!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Extract event images from coordinate data",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '-i', '--image', type=str, required=True,
        help="path to slide images directory")

    parser.add_argument(
        '-d', '--data', type=str, required=True,
        help="path to tab-delimited events data file with <frame_id>\t<x>\t<y>")

    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help="path to output hdf5 file including event both images and data")

    parser.add_argument(
        '-w', '--width', type=int, default=35,
        help="size of the event images to be cropped from slide images (odd)")

    parser.add_argument(
        '-m', '--mask', type=str, default=None,
        help="path to mask directory")

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
        '-v', '--verbose', action='count', default=0,
        help="verbosity level")

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

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    logger.info("Program started")
    logger.info(f"image directory:  {args.image}")
    logger.info(f"input data file:  {args.data}")
    logger.info(f"output file:      {args.output}")

    main(args)

    logger.info("Program finished successfully!")
