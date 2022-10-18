import os
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
        images = [cv2.imread(path, -1) for path in paths]
        self.image = cv2.merge(images)

    def writeImage(self, path):
        cv2.imwritemulti(self.image, path)
    
    def writeImageRGB(self, path):
        return None

    def showImage(self, channel, annotate=False):
        return None

    def readMask(self, path):
        return None

    def writeMask(self, path, name_format="Tile%06d.tif"):
        if self.mask is None:
            logger.error(f"mask {self.frame_id} is not loaded.")
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
        return(self.channels.index(ch))
