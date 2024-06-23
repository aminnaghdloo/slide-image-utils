from classes import Frame
import utils
import numpy as np
import pandas as pd
import cv2
import argparse


def get_noise_qc(frames, threshold):
    "count number of bright spots"
    frame_ids = []
    spot_count = []
    spot_area_mean = []
    spot_area_sdev = []

    for frame in frames:
        frame.readImage()
        mask = (frame.image > threshold).astype("uint8")
        num, _, stats, _ = cv2.connectedComponentsWithStats(mask, 4)
        area_mean = np.mean(stats[1:, cv2.CC_STAT_AREA]) if num > 1 else 0
        area_sdev = np.std(stats[1:, cv2.CC_STAT_AREA]) if num > 1 else 0
        frame_ids.append(frame.frame_id)
        spot_count.append(num - 1)
        spot_area_mean.append(area_mean)
        spot_area_sdev.append(area_sdev)

    df = pd.DataFrame(
        {
            "frame_id": frame_ids,
            "spot_count": spot_count,
            "area_mean": spot_area_mean,
            "area_dev": spot_area_sdev,
        }
    )

    return df


def count_spots(args):

    frames = []
    for i in range(args.nframes):
        frame_id = i + args.offset + 1
        paths = utils.generate_tile_paths(
            path=args.input,
            frame_id=frame_id,
            starts=[args.start],
            name_format=args.format,
        )

        frame = Frame(frame_id=frame_id, channels=[args.channel], paths=paths)
        frames.append(frame)

    df = get_noise_qc(frames, args.threshold)
    df.to_csv(args.output, index=False, sep="\t")

    print("Noise QC completed successfully!")


def main():
    # main inputs
    parser = argparse.ArgumentParser(
        description="Quality control of slide images with regard to noise.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="input path to slide images",
    )

    parser.add_argument(
        "-o", "--output", type=str, required=True, help="output file name"
    )

    parser.add_argument(
        "-c",
        "--channel",
        type=str,
        default="TRITC",
        help="target channel to analyze noise",
    )

    parser.add_argument(
        "-s", "--start", type=int, default=2305, help="channel start indices"
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=65000,
        help="noise intensity threshold",
    )

    parser.add_argument(
        "-f", "--offset", type=int, default=0, help="start frame offset"
    )

    parser.add_argument(
        "-n", "--nframes", type=int, default=2304, help="number of frames"
    )

    parser.add_argument(
        "-F",
        "--format",
        type=str,
        default="Tile%06d.tif",
        help="image name format",
    )

    args = parser.parse_args()

    count_spots(args)


if __name__ == "__main__":
    main()
