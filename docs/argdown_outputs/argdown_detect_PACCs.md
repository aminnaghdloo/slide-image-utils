## siu_detect_PACCs
### Usage
```
usage: siu_detect_PACCs [-h] -i INPUT -o OUTPUT [-m MASK_PATH] [-f OFFSET] [-n NFRAMES]
                        [-c CHANNELS [CHANNELS ...]] [-s STARTS [STARTS ...]]
                        [-F FORMAT] [-v] [-t THREADS] [--tophat_size TOPHAT_SIZE]
                        [--open_size OPEN_SIZE] [--blur_size BLUR_SIZE]
                        [--blur_sigma BLUR_SIGMA] [--thresh_size THRESH_SIZE]
                        [--thresh_offsets THRESH_OFFSETS [THRESH_OFFSETS ...]]
                        [--min_seed_dist MIN_SEED_DIST]
                        [--nucleus_channel NUCLEUS_CHANNEL]
                        [--mask_channels MASK_CHANNELS [MASK_CHANNELS ...]]
                        [--exclude_border] [--include_edge_frames] [--extract_images]
                        [-w WIDTH] [--mask_flag] [--sort SORT SORT]
                        [--filter FILTER FILTER FILTER]
```
### Arguments
#### Quick reference table
|Short|Long                   |Default                           |Description                                                                                                                                                                |
|-----|-----------------------|----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`-h` |`--help`               |                                  |show this help message and exit                                                                                                                                            |
|`-i` |`--input`              |`None`                            |input path to slide images                                                                                                                                                 |
|`-o` |`--output`             |`None`                            |output path                                                                                                                                                                |
|`-m` |`--mask_path`          |`None`                            |mask path to save frame masks if needed                                                                                                                                    |
|`-f` |`--offset`             |`0`                               |start frame offset                                                                                                                                                         |
|`-n` |`--nframes`            |`2304`                            |number of frames                                                                                                                                                           |
|`-c` |`--channels`           |`['DAPI', 'TRITC', 'CY5', 'FITC']`|channel names                                                                                                                                                              |
|`-s` |`--starts`             |`[1, 2305, 4609, 9217]`           |channel start indices                                                                                                                                                      |
|`-F` |`--format`             |`Tile%06d.tif`                    |image name format                                                                                                                                                          |
|`-v` |`--verbose`            |`0`                               |verbosity level                                                                                                                                                            |
|`-t` |`--threads`            |`0`                               |number of threads for parallel processing                                                                                                                                  |
|     |`--tophat_size`        |`45`                              |TopHat filter kernel size                                                                                                                                                  |
|     |`--open_size`          |`5`                               |Open morphological filter kernel size                                                                                                                                      |
|     |`--blur_size`          |`5`                               |Gaussian blur filter kernel size                                                                                                                                           |
|     |`--blur_sigma`         |`2`                               |Gaussian blur filter parameter sigma                                                                                                                                       |
|     |`--thresh_size`        |`25`                              |adaptive threshold window size                                                                                                                                             |
|     |`--thresh_offsets`     |`[-2000, -2000, -2000, -2000]`    |adaptive threshold offset values for channels (negative)                                                                                                                   |
|     |`--min_seed_dist`      |`7`                               |minimum allowed distance [pixels] between each pair of seeds                                                                                                               |
|     |`--nucleus_channel`    |`DAPI`                            |channel name to use as seed                                                                                                                                                |
|     |`--mask_channels`      |`['DAPI', 'TRITC', 'CY5', 'FITC']`|channels to segment                                                                                                                                                        |
|     |`--exclude_border`     |                                  |exclude events that are on image borders                                                                                                                                   |
|     |`--include_edge_frames`|                                  |include frames that are on the edge of slide                                                                                                                               |
|     |`--extract_images`     |                                  |extract images of detected events and output hdf5 file                                                                                                                     |
|`-w` |`--width`              |`35`                              |
		size of the event images to be cropped from slide images (odd).
		Works only when --extract_images is set.
		                                                           |
|     |`--mask_flag`          |                                  |store event masks when extracting images                                                                                                                                   |
|     |`--sort`               |`[]`                              |
		sort events based on feature values.

		Usage:	  <command> --sort <feature> <order>
		Example:	<command> --sort TRITC_mean I
		order:	  I: Increasing / D: Decreasing
		|
|     |`--filter`             |`[]`                              |
		feature range for filtering detected events.

		Usage:	  <command> --feature_range <feature> <min> <max>
		Example:	<command> --feature_range DAPI_mean 0 10000
		      |

#### `-h`, `--help`
show this help message and exit

#### `-i`, `--input` (Default: None)
input path to slide images

#### `-o`, `--output` (Default: None)
output path

#### `-m`, `--mask_path` (Default: None)
mask path to save frame masks if needed

#### `-f`, `--offset` (Default: 0)
start frame offset

#### `-n`, `--nframes` (Default: 2304)
number of frames

#### `-c`, `--channels` (Default: ['DAPI', 'TRITC', 'CY5', 'FITC'])
channel names

#### `-s`, `--starts` (Default: [1, 2305, 4609, 9217])
channel start indices

#### `-F`, `--format` (Default: Tile%06d.tif)
image name format

#### `-v`, `--verbose` (Default: 0)
verbosity level

#### `-t`, `--threads` (Default: 0)
number of threads for parallel processing

#### `--tophat_size` (Default: 45)
TopHat filter kernel size

#### `--open_size` (Default: 5)
Open morphological filter kernel size

#### `--blur_size` (Default: 5)
Gaussian blur filter kernel size

#### `--blur_sigma` (Default: 2)
Gaussian blur filter parameter sigma

#### `--thresh_size` (Default: 25)
adaptive threshold window size

#### `--thresh_offsets` (Default: [-2000, -2000, -2000, -2000])
adaptive threshold offset values for channels (negative)

#### `--min_seed_dist` (Default: 7)
minimum allowed distance [pixels] between each pair of seeds

#### `--nucleus_channel` (Default: DAPI)
channel name to use as seed

#### `--mask_channels` (Default: ['DAPI', 'TRITC', 'CY5', 'FITC'])
channels to segment

#### `--exclude_border`
exclude events that are on image borders

#### `--include_edge_frames`
include frames that are on the edge of slide

#### `--extract_images`
extract images of detected events and output hdf5 file

#### `-w`, `--width` (Default: 35)
                 size of the event images to be cropped from slide images
(odd).                 Works only when --extract_images is set.

#### `--mask_flag`
store event masks when extracting images

#### `--sort` (Default: [])
                 sort events based on feature values.                  Usage:
<command> --sort <feature> <order>                 Example:        <command>
--sort TRITC_mean I                 order:    I: Increasing / D: Decreasing

#### `--filter` (Default: [])
                 feature range for filtering detected events.
Usage:    <command> --feature_range <feature> <min> <max>
Example:        <command> --feature_range DAPI_mean 0 10000


