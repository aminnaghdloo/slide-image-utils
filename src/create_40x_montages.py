from PIL import Image
from classes import Frame
import pandas as pd
import numpy as np
import cv2
import h5py
import os
import sys
import argparse
import utils


def channels2montage(images, b_index, g_index, r_index, order_index):
    "Create montages from list of images."
    bgr = utils.channels_to_bgr(images, b_index, g_index, r_index)
    gray = np.concatenate([images[:,:,:,k] for k in order_index], axis=2)
    gray = np.stack([gray] * 3, axis=3)
    montages = np.concatenate([bgr, gray], axis=2)
    return montages


def main(args):
    
    # inputs
    input       = args.input
    data_path   = args.data
    output      = args.output
    width       = args.width
    channels    = args.channels
    gain        = args.gain
    red         = args.red
    green       = args.green
    blue        = args.blue
    order       = args.order
    verbosity   = args.verbose
    
    logger = utils.get_logger(__name__, verbosity)

    # reading input data
    df = pd.read_table(data_path, sep='\t')
    
    images = []
    index_to_drop = []
    for index, row in df.iterrows():
        paths = [f"{input}/{row.cell_id}-{row.frame_id}-{row.x}-{row.y}-{c}.tif"
                for c in channels]
        
        if not all([os.path.isfile(path) for path in paths]):
            index_to_drop.append(index)
            continue

        image = np.stack([np.array(Image.open(path)) for path in paths], axis=2)
        image = np.fliplr(np.flipud(image))

        h, w, _ = image.shape
        crop = image[(h // 2 - width // 2):(h // 2 + width // 2 + 1),
                     (w // 2 - width // 2):(w // 2 + width // 2 + 1), :]
        images.append(crop)
        logger.info(f"Loaded image {index}")

    # create montages
    images = np.stack(images, axis=0)
    images = utils.apply_gain(images, gain)
    df.drop(index_to_drop, inplace=True)
    df.reset_index(inplace=True)

    logger.info("Creating the montages...")
    r_index = [channels.index(channel) for channel in red]
    g_index = [channels.index(channel) for channel in green]
    b_index = [channels.index(channel) for channel in blue]
    order_index = [channels.index(channel) for channel in order]
    montages = channels2montage(images, b_index, g_index, r_index, order_index)

    montages = utils.convert_dtype(montages, 'uint8')
    #montages = (montages.astype(float) // 256).astype('uint8')

    if args.separate:
        output_path = f"{os.path.dirname(output)}/" \
                      f"{os.path.basename(output).split('.')[0]}"
        os.makedirs(output_path, exist_ok=True)
        for i, row in df.iterrows():
            temp_path = f"{output_path}/{int(row.cell_id)}-{int(row.frame_id)}"\
                        f"-{int(row.x)}-{int(row.y)}_40X.jpg"
            cv2.imwrite(temp_path, montages[i])
            logger.info(f"Created {temp_path}")
    else:
        cv2.imwritemulti(output, montages)

    logger.info('Finished creating the montages!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Create montages of events from 40x frame images.",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '-i', '--input', type=str, required=True,
        help="path to 40x image directory containing 40x frame images")

    parser.add_argument(
        '-d', '--data', type=str, default=None,
        help="path to tab-delimited reimaged data file")

    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help="path to output gallery image")

    parser.add_argument(
        '-w', '--width', type=int, default=35,
        help="size of each event image crop (odd)")
    
    parser.add_argument(
        '-c', '--channels', nargs='+', default=['DAPI', 'TRITC', 'CY5', 'FITC'],
        help="channel(s) to include while reading images")

    parser.add_argument(
        '-g', '--gain', nargs='+', type=float, default=[1,1,1,1],
        help="gains applied to each channel")

    parser.add_argument(
        '-R', '--red', nargs='+', default=[],
        help="channel(s) to be shown in red color")

    parser.add_argument(
        '-G', '--green', nargs='+', default=[],
        help="channel(s) to be shown in green color")
    
    parser.add_argument(
        '-B', '--blue', nargs='+', default=[],
        help="channel(s) to be shown in blue color")

    parser.add_argument(
        '-O', '--order', nargs='+', default=['DAPI', 'TRITC', 'FITC', 'CY5'],
        help="order of channels in grayscale section of the montage")

    parser.add_argument(
        '-s', '--separate', action='store_true', default=False,
        help="""
        save montages individually for each event as 
        <cell_id>-<frame_id>-<x>-<y>_40x.jpg"""
    )

    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help="verbosity level")

    args = parser.parse_args()
    logger = utils.get_logger("parse_args", args.verbose)

    # checking for potential errors in input arguments
    if not os.path.exists(args.input):
        logger.error(f"image data file {args.input} not found!")
        sys.exit(-1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    logger.info("Program started")
    logger.info(f"input image dir:  {args.input}")
    logger.info(f"input data file:  {args.data}")
    logger.info(f"output file:      {args.output}")

    main(args)

    logger.info("Program finished successfully!")
