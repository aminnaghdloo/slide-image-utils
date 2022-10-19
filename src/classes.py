import os
import sys
import cv2
import numpy as np
import utils

logger = utils.get_logger(__name__, 20)

class Frame:
    """
    A class for frame image data of a slide.

    """
    def __init__(self, frame_id, channels):
        self.frame_id = frame_id
        self.channels = channels
        self.image = None
        self.mask = None

    def readImage(self, paths):
        if not os.path.exists(paths[0]):
            print(paths[0])
            logger.error("image is not found!")
            sys.exit(-1)
        else:
            images = [cv2.imread(path, -1) for path in paths]
            
            ### reading CSI-Cancer compressed images
            suffix = paths[0].split('.')[-1]
            tags = [path.replace(f'.{suffix}', '.tags') for path in paths]

            if suffix == 'jpg' and os.path.exists(tags[0]):
                vals = utils.readPreservedMinMax(tags)
                for i in range(len(images)):
                    a = (vals['maxval'][i] - vals['minval'][i])
                    b = vals['minval'][i]
                    images[i] = images[i].astype('float')
                    images[i] = a * images[i] + b
                    images[i] = images[i].astype('uint16')
            ### end of reading compressed images

            self.image = cv2.merge(images)


    def writeImage(self, path):
        cv2.imwritemulti(self.image, path)
    
    def writeImageRGB(self, path):
        return None

    def showImage(self, channel, annotate=False):
        return None

    def readMask(self, path, name_format="Tile%06d.tif"):
        return None

    def writeMask(self, path, name_format="Tile%06d.tif"):
        if self.mask is None:
            logger.error(f"mask is not loaded!")
            sys.exit(-1)
        else:
            os.makedirs(path, exist_ok=True)
            cv2.imwrite(
                f"{path}/{name_format}" % self.frame_id,
                self.mask
            )

    def showMask(self, annotate=False):
        return None
    
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

    def extract_crops(self, data, width, mask_flag=False):
        "Extracts event image crops from frame images"
        if self.image is None:
            logger.error("frame image is not loaded!")
            sys.exit(-1)

        elif not mask_flag:
            edge = round((width - 1) / 2)
            n = len(data)
            x = data['x'].astype(int) + edge
            y = data['y'].astype(int) + edge
            
            pad = np.zeros(len(self.channels))
            image = cv2.copyMakeBorder(
                self.image, edge, edge, edge, edge, cv2.BORDER_CONSTANT, pad)

            indices = data.index.tolist()
            out = np.zeros((n, width, width, len(self.channels)),dtype='uint16')

            for i, index in enumerate(indices):
                out[i, ...] = image[(x[index] - edge):(x[index] + edge + 1),
                                    (y[index] - edge):(y[index] + edge + 1), :]

            return(out)

        elif self.mask is None:
            logger.error("frame mask is not loaded")
            sys.exit(-1)

        else:
            edge = round((width - 1) / 2)
            n = len(data)
            x = data['x'].astype(int) + edge
            y = data['y'].astype(int) + edge
            
            mask = cv2.copyMakeBorder(
                self.mask, edge, edge, edge, edge, cv2.BORDER_CONSTANT, 0)
            mask = mask[..., np.newaxis]

            pad = np.zeros(len(self.channels))
            image = cv2.copyMakeBorder(
                self.image, edge, edge, edge, edge, cv2.BORDER_CONSTANT, pad)
            
            ids = mask[x - 1, y - 1].flatten()
            indices = data.index.tolist()
            out = np.zeros((n, width, width, len(self.channels)),dtype='uint16')

            for i, index, id in zip(range(n), indices, ids):
                scimage = np.multiply(image, (mask == id).astype('uint16'))
                out[i, ...] = scimage[(x[index] - edge):(x[index] + edge + 1),
                                      (y[index] - edge):(y[index] + edge + 1),:]

            return(out)
