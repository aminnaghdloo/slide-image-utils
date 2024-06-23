import cv2
import argparse
import glob
import os
import sys


def readPreservedMinMax(meta_names):
    "This function reads preserved min and max pixel values for JPEG images."
    minval = []
    maxval = []
    for i in range(len(meta_names)):
        with open(meta_names[i]) as file:
            lines = file.read().splitlines()
            for line in lines:
                tag, val = line.split("=")

                if tag == "PreservedMinValue":
                    minval.append(int(float(val) * 256))

                elif tag == "PreservedMaxValue":
                    maxval.append(int(float(val) * 256))

    vals = {"minval": minval, "maxval": maxval}
    return vals


def main(args):
    frame_names = glob.glob(args.input + "/*.jpg")
    meta_names = [item.replace(".jpg", ".tags") for item in frame_names]
    file_exists = [os.path.isfile(meta_name) for meta_name in meta_names]

    if not all(file_exists):
        sys.exit("All JPEG images need to have corresponding TAG files!")

    vals = readPreservedMinMax(meta_names)

    for i, frame_name in enumerate(frame_names):
        image = cv2.imread(frame_name, -1)
        a = vals["maxval"][i] - vals["minval"][i]
        b = vals["minval"][i]
        image = image.astype("float")
        image = a * image + b
        image[image > 65535] = 65535
        image = image.astype("uint16")
        outfile = (
            args.output + "/" + os.path.basename(frame_name).replace(
			".jpg", ".tif"
			)
        )
        cv2.imwrite(outfile, image)
        if args.verbose:
            print(f"Generated {outfile}")

    if args.verbose:
        print(f"Converted slide images successfully!")


if __name__ == "__main__":

    # main inputs
    parser = argparse.ArgumentParser(
        description="Convert JPG frame images to TIF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="input directory with JPG frame images and their corresponding tags",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="output directory to save TIF frame images",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="verbose execution",
    )

    args = parser.parse_args()

    main(args)
