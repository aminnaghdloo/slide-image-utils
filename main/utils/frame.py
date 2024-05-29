from skimage import measure, color
import pandas as pd
import numpy as np
import os
import sys
import cv2
import utils

logger = utils.get_logger(__name__, 20)

class Frame:
    """
    A class for frame image data of a slide.

    """
    def __init__(self, frame_id, channels, paths):
        self.frame_id = frame_id
        self.channels = channels
        self.paths = paths
        self.image = None
        self.mask = None

    def readImage(self):
        if not os.path.exists(self.paths[0]):
            print(self.paths[0])
            logger.debug(f"paths[0]: {self.paths[0]}")
            logger.error("frame image does not exist!")
            sys.exit(-1)
        else:
            images = [cv2.imread(path, -1) for path in self.paths]
            
            ### reading CSI-Cancer compressed images
            suffix = self.paths[0].split('.')[-1]
            if suffix == 'jpg':
                tags = [path.replace(f'.{suffix}', '.tags') for path in self.paths]

                if os.path.exists(tags[0]):
                    vals = utils.readPreservedMinMax(tags)
                    for i in range(len(images)):
                        a = (vals['maxval'][i] - vals['minval'][i])
                        b = vals['minval'][i]
                        images[i] = images[i].astype('float')
                        images[i] = a * images[i] + b
                        images[i][images[i] > 65535] = 65535
                        images[i] = images[i].astype('uint16')
            ### end of reading compressed images
            self.image = cv2.merge(images)

    def readMask(self, mask_dir, name_format="Tile%06d.tif"):
        mask_path = f"{mask_dir}/{name_format}" % self.frame_id
        if not os.path.exists(mask_path):
            logger.error(f"mask {mask_path} does not exist!")
            sys.exit(-1)
        else:
            self.mask = cv2.imread(mask_path, -1)

    def writeMask(self, mask_dir, name_format="Tile%06d.tif",
                  in_mask=None):
        mask = self.mask if in_mask is None else in_mask
        if mask is None:
            logger.error(f"frame mask is not loaded!")
            sys.exit(-1)
        else:
            os.makedirs(mask_dir, exist_ok=True)
            cv2.imwrite(
                f"{mask_dir}/{name_format}" % self.frame_id,
                mask
            )
    
    def is_edge(self):
        if (
            self.frame_id <= 48 or
            self.frame_id > 2256 or
            self.frame_id % 24 in [0, 1, 2, 23]
        ):
            return(True)
        else:
            return(False)
        
    def get_ch(self, ch):
        "Get channel index for a given channel name"
        if ch not in self.channels:
            logger.error("Channel does not exist in this frame")
            sys.exit(-1)
        else:
            return(self.channels.index(ch))

    def calc_basic_features(self, in_mask=None):
        "Extract basic features of events from frame image."
        mask = self.mask if in_mask is None else in_mask
        if self.image is None:
            logger.error("frame image is not loaded!")
            sys.exit(-1)
        elif mask is None:
            logger.error("frame mask is not loaded!")
            sys.exit(-1)
        else:
            props = measure.regionprops_table(
                mask, self.image, separator='_',
                properties=[
                    'label', 'centroid', 'area', 'eccentricity',
                    'intensity_mean']
            )
            props = pd.DataFrame(props)
            colnames = ['cell_id', 'y', 'x', 'area', 'eccentricity']
            colnames.extend([ch + '_mean' for ch in self.channels])
            props.set_axis(colnames, axis=1, inplace=True)
            props = props.astype({'cell_id': int})
            props.insert(0, 'frame_id', self.frame_id)
        return(props)
 
    def calc_morph_features(self):
        "Extract basic features of events from frame image."
        if self.image is None:
            logger.error("frame image is not loaded!")
            sys.exit(-1)
        elif self.mask is None:
            logger.error("frame mask is not loaded!")
            sys.exit(-1)
        else:
            props = measure.regionprops_table(
                self.mask, self.image, separator='_',
                properties=[
                    'label', 'centroid', 'area', 'eccentricity',
                    'axis_major_length', 'axis_minor_length',
                    'equivalent_diameter_area', 'extent', 'feret_diameter_max',
                    'perimeter', 'intensity_mean']
            )
            props = pd.DataFrame(props)
            colnames = ['cell_id', 'y', 'x', 'area', 'eccentricity',
                        'major_axis', 'minor_axis', 'diameter','ratio_bb',
                        'feret_diameter', 'perimeter']
            colnames.extend([ch + '_mean' for ch in self.channels])
            props.set_axis(colnames, axis=1, inplace=True)
            props = props.astype({'cell_id': int})
            props.insert(0, 'frame_id', self.frame_id)
        return(props)

    def calc_event_features(self, func, channel, columns):
        "Extract desired features of events from frame image."
        if self.image is None:
            logger.error("frame image is not loaded!")
            sys.exit(-1)
        elif self.mask is None:
            logger.error("frame mask is not loaded!")
            sys.exit(-1)
        else:
            props = measure.regionprops_table(
                        self.mask,
                        self.image[..., self.get_ch(channel)],
                        properties=('label', 'centroid'),
                        extra_properties=[func])
            props = pd.DataFrame(props)
            colnames = ['cell_id', 'y', 'x']
            colnames.extend([channel + '_' + c for c in columns])
            props.set_axis(colnames, axis=1, inplace=True)
            props = props.astype({'cell_id': int})
            props.insert(0, 'frame_id', self.frame_id)
            return(props)

    def extract_crops(self, data, width, mask_flag=False):
        "Extracts event image crops from frame images"
        if self.image is None:
            logger.error("frame image is not loaded!")
            sys.exit(-1)

        elif mask_flag and self.mask is None:
            logger.error("frame mask is not loaded!")
            sys.exit(-1)

        else:
            edge = round((width - 1) / 2)
            n = len(data)
            x = data['x'].astype(int) + edge
            y = data['y'].astype(int) + edge

            if n == 0:
                # logger.warning("Empty data file!")
                return(None, None)
            
            indices = data.index.tolist()
            out_image = np.zeros(
                (n, width, width, len(self.channels)),dtype='uint16')

            pad = np.zeros(len(self.channels))
            image = cv2.copyMakeBorder(
                self.image, edge, edge, edge, edge, cv2.BORDER_CONSTANT, pad)
            
            if mask_flag:
                out_mask = np.zeros((n, width, width, 1), dtype='uint16')
                mask = cv2.copyMakeBorder(
                    self.mask, edge, edge, edge, edge, cv2.BORDER_CONSTANT, 0)
                mask = mask[..., np.newaxis]
            
                for i, index in enumerate(indices):
                    out_mask[i, ...] = mask[
                                        (y[index] - edge):(y[index] + edge + 1),
                                        (x[index] - edge):(x[index] + edge + 1),
                                        :]
                    out_image[i, ...] = image[
                                        (y[index] - edge):(y[index] + edge + 1),
                                        (x[index] - edge):(x[index] + edge + 1),
                                        :]

                return(out_image, out_mask)

            else:
                for i, index in enumerate(indices):
                    out_image[i, ...] = image[
                                        (y[index] - edge):(y[index] + edge + 1),
                                        (x[index] - edge):(x[index] + edge + 1),
                                        :]
                return(out_image, None)

    def writeImage(self, path):
        cv2.imwritemulti(self.image, path)
    
    def writeImageRGB(self, path):
        return None

    def showImage(self, channel, annotate=False):
        return None

    def showMask(self, annotate=False):
        return None
