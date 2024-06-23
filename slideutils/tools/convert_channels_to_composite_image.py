import cv2
import numpy as np
import argparse
import os
import sys
from slideutils.utils import utils

def main():
	parser = argparse.ArgumentParser(
		description="Create composite images from gray scale image channels",
		formatter_class=argparse.RawTextHelpFormatter
	)
	parser.add_argument(
		"-i", "--input", required=True, type=str, nargs="+",
		help="path(s) to gray-scale channel images"
	)
	parser.add_argument(
		"-B", "--blue", required=True, type=int, nargs="+",
		help="order indices of images to be shown in blue color"
	)
	parser.add_argument(
		"-G", "--green", required=True, type=int, nargs="+",
		help="order indices of images to be shown in green color"
	)
	parser.add_argument(
		"-R", "--red", required=True, type=int, nargs="+",
		help="order indices of images to be shown in red color"
	)
	parser.add_argument(
		"-o", "--output", required=True, type=str,
		help="path to output color image"
	)

	args = parser.parse_args()
	for item in args.input:
		if not os.path.exists(item):
			print(f"image file {item} not found!")
			sys.exit(-1)
	os.makedirs(os.path.dirname(args.output), exist_ok=True)

	image = np.stack([cv2.imread(item, -1) for item in args.input], axis=-1)
	output = utils.channels_to_bgr(image, args.blue, args.green, args.red)
	cv2.imwrite(args.output, output)

	print(f"Successfully converted channel images to {args.output}!")

if __name__ == "__main__":
	main()
