import logging
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure


def get_logger(module_name, verbosity_level):
    """
    Creates a logger object for console.
    
    Parameters
    ----------
    module_name : str
        Name of the module/program that is logging
    verbosity_level : int (choices: 0, 1, 2)
        Level of verbosity requested by the module/program
    
    Returns
    -------
    Logger
        Logger object from logging library

    """
    logger = logging.getLogger(module_name)
    level = (30 - 10 * min(verbosity_level, 2))
    logger.setLevel(level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "[%(levelname)s] @ %(asctime)s : %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return(logger)


def generate_tile_paths(path, frame_id, starts, name_format):
    paths = [f"{path}/{name_format}" % (frame_id + j - 1) for j in starts]
    return(paths)


def is_edge(frame_id):
        if (
            frame_id <= 48 or
            frame_id > 2256 or
            frame_id % 24 in [0, 1, 2, 23]
        ):
            return(True)
        else:
            return(False)


def fill_holes(input_mask):
    "Fill the holes in masks."
    assert input_mask.dtype == np.uint8
    h, w = input_mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:h + 1, 1:w + 1] = input_mask.copy()
    canvas[canvas != 0] = 1
    mask = np.zeros((h + 4, w + 4), np.uint8)
    mask[1:h + 3, 1:w + 3] = canvas
    cv2.floodFill(canvas, mask, (0, 0), 1)
    canvas = 1 - canvas[1:h + 1, 1:w + 1]
    return(canvas * input_mask.max() + input_mask)


def pltImageShow(img, title='title', size=(8,6), dpi=150, out=None):
    "Display image using matplotlib.pyplot package."
    if(img.dtype == 'uint16'):
        img = img.astype('float')
        img = img / 65535
    plt.figure(figsize=size,dpi=dpi)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(title)
    if out is not None:
        plt.savefig(out)
    plt.show()
    
    
def pltImageShow2by2(img, title='title', size=(8,6), dpi=150, out=None):
    "Display image using matplotlib.pyplot package."
    if(img.dtype == 'uint16'):
        img = img.astype('float')
        img = img / 65535
    h, w, d = img.shape
    assert d == 4
    row1 = np.concatenate((img[..., 0], img[..., 1]), axis=0)
    row2 = np.concatenate((img[..., 2], img[..., 3]), axis=0)
    image = np.concatenate((row1, row2), axis=1)
    plt.figure(figsize=size,dpi=dpi)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(title)
    if out is not None:
        plt.savefig(out)
    plt.show()
    

def calc_basic_features(frame):
    "Extract cell info of the frame image based on the mask."
    props = measure.regionprops_table(
        frame.mask, frame.image, separator='_',
        properties=[
            'label', 'centroid', 'area', 'eccentricity',
            'intensity_mean']
    )
    props = pd.DataFrame(props)
    colnames = ['cell_id', 'x', 'y', 'area', 'eccentricity']
    colnames.extend([ch + '_mean' for ch in frame.channels])
    props.set_axis(colnames, axis=1, inplace=True)
    props = props.astype({'cell_id': int})
    props.insert(0, 'frame_id', frame.frame_id)
    return(props)


def filter_events(features, filters, verbosity):
    "Filter detected events before saving the results"

    logger = get_logger('filter_events', verbosity)

    n = len(features)
    sel = pd.DataFrame({'index' : [True for i in range(n)]})

    
    for filter in filters:
        f_name = filter[0]
        f_min = float(filter[1])
        f_max = float(filter[2])

        if f_name not in features.columns:
            logger.warning(f"Cannot filter on {f_name}: Feature not found!")
            continue
        else:
            sel['index'] = sel['index'] &\
                (features[f_name] >= f_min) &\
                (features[f_name] <= f_max)

    features = features[sel['index']]
    logger.info(f"Filtered {n} events down to {len(features)} events")

    return(features)


def readPreservedMinMax(meta_names):
    "This function reads preserved min and max pixel values for JPEG images."
    minval = []
    maxval = []
    for i in range(len(meta_names)):
        with open(meta_names[i]) as file:
            lines = file.read().splitlines()
            for line in lines:
                tag, val = line.split('=')
                
                if tag == 'PreservedMinValue':
                    minval.append(int(float(val) * 256))
                    
                elif tag == 'PreservedMaxValue':
                    maxval.append(int(float(val) * 256))
    
    vals = {'minval':minval, 'maxval':maxval}
    return(vals)
