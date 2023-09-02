import cv2
import sys
import os
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "firefox"

import numpy as np

def main():
    if len(sys.argv) != 3:
        quit('python visualize_image.py <image.tif> <intensity_shift>')
    else:
        image_name = sys.argv[1]
        shift = int(sys.argv[2])
        x = cv2.imread(image_name, -1)
        #print(x.dtype, x.shape)
        x = x + shift
        fig = px.imshow(x)
        fig.show()

        
        #cv2.imshow(os.path.basename(image_name), x)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
