## siu_detect_LEVs
### Usage
```
usage: argdown [-h] -i INPUT -o OUTPUT [-m MASK] [-f OFFSET] [-n NFRAMES]
               [-c CHANNELS [CHANNELS ...]]
               [-C BG_CELL_CHANNELS [BG_CELL_CHANNELS ...]]
               [-s STARTS [STARTS ...]] [-F FORMAT] [-t THREADS]
               [-T TARGET_CHANNEL] [-L LOW] [-H HIGH] [-k KERNEL]
               [--max_val MAX_VAL] [-v] [--include_edge_frames]
               [--selected_frames [SELECTED_FRAMES ...]] [--extract_images]
               [-w WIDTH] [--mask_flag] [--sort SORT SORT]
               [--filter FILTER FILTER FILTER]
```
### Arguments
#### Quick reference table
|Short|Long                   |Default                           |Description                                                                                                                                                                                                                                                                                                                                                              |
|-----|-----------------------|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`-h` |`--help`               |                                  |show this help message and exit                                                                                                                                                                                                                                                                                                                                          |
|`-i` |`--input`              |`None`                            |path to slide images directory                                                                                                                                                                                                                                                                                                                                           |
|`-o` |`--output`             |`None`                            |output file path                                                                                                                                                                                                                                                                                                                                                         |
|`-m` |`--mask`               |`None`                            |path to a directory to save event masks [optional]                                                                                                                                                                                                                                                                                                                       |
|`-f` |`--offset`             |`0`                               |start frame offset                                                                                                                                                                                                                                                                                                                                                       |
|`-n` |`--nframes`            |`2304`                            |number of frames to process                                                                                                                                                                                                                                                                                                                                              |
|`-c` |`--channels`           |`['DAPI', 'TRITC', 'CY5', 'FITC']`|channel names                                                                                                                                                                                                                                                                                                                                                            |
|`-C` |`--bg_cell_channels`   |`['DAPI', 'CY5']`                 |name of channels for which background cells are positive                                                                                                                                                                                                                                                                                                                 |
|`-s` |`--starts`             |`[1, 2305, 4609, 9217]`           |channel start indices                                                                                                                                                                                                                                                                                                                                                    |
|`-F` |`--format`             |`Tile%06d.tif`                    |image name format                                                                                                                                                                                                                                                                                                                                                        |
|`-t` |`--threads`            |`0`                               |number of threads for parallel processing                                                                                                                                                                                                                                                                                                                                |
|`-T` |`--target_channel`     |`TRITC`                           |target channel name for LEV detection                                                                                                                                                                                                                                                                                                                                    |
|`-L` |`--low`                |`99.7`                            |low threshold for segmentation [percentile]                                                                                                                                                                                                                                                                                                                              |
|`-H` |`--high`               |`2`                               |high threshold for segmentation [ratio-to-median]                                                                                                                                                                                                                                                                                                                        |
|`-k` |`--kernel`             |`75`                              |size of tophat filter kernel                                                                                                                                                                                                                                                                                                                                             |
|     |`--max_val`            |`65535`                           |maximum pixel value for foreground during thresholding                                                                                                                                                                                                                                                                                                                   |
|`-v` |`--verbose`            |`0`                               |verbosity level                                                                                                                                                                                                                                                                                                                                                          |
|     |`--include_edge_frames`|                                  |include frames that are on the edge of slide                                                                                                                                                                                                                                                                                                                             |
|     |`--selected_frames`    |`[]`                              |list of selected frames to be processed                                                                                                                                                                                                                                                                                                                                  |
|     |`--extract_images`     |                                  |extract images of detected events and output hdf5 file                                                                                                                                                                                                                                                                                                                   |
|`-w` |`--width`              |`45`                              |
		size of the event images to be cropped from slide images (odd).
		Works only when --extract_images is set.
		                                                                                                                                                                                                                                                         |
|     |`--mask_flag`          |                                  |store event masks when extracting images                                                                                                                                                                                                                                                                                                                                 |
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

		Acceptable thresholds are listed in the following table:

		feature		 minimum	 maximum
		-------		 -------	 -------
		area			0		   +inf
		eccentricity	0		   1
		<channel>_mean  0		   <MAX_VAL>
		|

#### `-h`, `--help`
show this help message and exit

#### `-i`, `--input` (Default: None)
path to slide images directory

#### `-o`, `--output` (Default: None)
output file path

#### `-m`, `--mask` (Default: None)
path to a directory to save event masks [optional]

#### `-f`, `--offset` (Default: 0)
start frame offset

#### `-n`, `--nframes` (Default: 2304)
number of frames to process

#### `-c`, `--channels` (Default: ['DAPI', 'TRITC', 'CY5', 'FITC'])
channel names

#### `-C`, `--bg_cell_channels` (Default: ['DAPI', 'CY5'])
name of channels for which background cells are positive

#### `-s`, `--starts` (Default: [1, 2305, 4609, 9217])
channel start indices

#### `-F`, `--format` (Default: Tile%06d.tif)
image name format

#### `-t`, `--threads` (Default: 0)
number of threads for parallel processing

#### `-T`, `--target_channel` (Default: TRITC)
target channel name for LEV detection

#### `-L`, `--low` (Default: 99.7)
low threshold for segmentation [percentile]

#### `-H`, `--high` (Default: 2)
high threshold for segmentation [ratio-to-median]

#### `-k`, `--kernel` (Default: 75)
size of tophat filter kernel

#### `--max_val` (Default: 65535)
maximum pixel value for foreground during thresholding

#### `-v`, `--verbose` (Default: 0)
verbosity level

#### `--include_edge_frames`
include frames that are on the edge of slide

#### `--selected_frames` (Default: [])
list of selected frames to be processed

#### `--extract_images`
extract images of detected events and output hdf5 file

#### `-w`, `--width` (Default: 45)
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
Acceptable thresholds are listed in the following table:
feature          minimum         maximum                 -------
-------         -------                 area                    0
+inf                 eccentricity    0                  1
<channel>_mean  0                  <MAX_VAL>


