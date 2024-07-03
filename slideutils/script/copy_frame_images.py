from slideutils.utils import utils
import os
import shutil
import argparse

def main():
    # main inputs
    parser = argparse.ArgumentParser(
        description="Copy frame images to a different directory.",
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
        "-o", "--output", type=str, required=True, help="output path"
    )

    parser.add_argument(
        "-f", "--offset", type=int, default=0, help="start frame offset"
    )

    parser.add_argument(
        "-n", "--nframes", type=int, default=2304, help="number of frames"
    )

    parser.add_argument(
        "-s",
        "--starts",
        type=int,
        nargs="+",
        default=[1, 2305, 4609, 9217],
        help="channel start indices",
    )

    parser.add_argument(
        "-F",
        "--format",
        type=str,
        nargs="+",
        default=["Tile%06d.tif"],
        help="image name format",
    )

    parser.add_argument(
        "--selected_frames",
        type=int,
        nargs="*",
        default=[],
        help="list of selected frames to be processed",
    )

    args = parser.parse_args()

    # check if there is a selection of frames to process
    if args.selected_frames:
        frame_ids = args.selected_frames
    else:
        frame_ids = [i + args.offset + 1 for i in range(args.nframes)]

    files2copy = []
    for frame_id in frame_ids:
        paths = utils.generate_tile_paths(
            path=args.input,
            frame_id=frame_id,
            starts=args.starts,
            name_format=args.format,
        )
        files2copy.extend(paths)

    for file_path in files2copy:
        shutil.copy2(file_path, args.output + '/' + os.path.basename(file_path))


if __name__ == "__main__":
    main()
