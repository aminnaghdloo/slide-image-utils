## siu_extract_morphs
### Usage
```
usage: argdown [-h] -i INPUT -o OUTPUT -m MASK [-f OFFSET] [-n NFRAMES]
               [-c CHANNELS [CHANNELS ...]] [-s STARTS [STARTS ...]]
               [-F FORMAT] [-t THREADS] [-T TARGET_CHANNEL] [-L LOW] [-H HIGH]
               [-k KERNEL] [--max_val MAX_VAL] [-v] [--include_edge_frames]
               [--filter FILTER FILTER FILTER]
```
### Arguments
#### Quick reference table
|Short|Long                   |Default                           |Description                                                                                                                                                                                                                                                                                                                                                                    |
|-----|-----------------------|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`-h` |`--help`               |                                  |show this help message and exit                                                                                                                                                                                                                                                                                                                                                |
|`-i` |`--input`              |`None`                            |path to slide images directory                                                                                                                                                                                                                                                                                                                                                 |
|`-o` |`--output`             |`None`                            |output file path                                                                                                                                                                                                                                                                                                                                                               |
|`-m` |`--mask`               |`None`                            |path to slide image masks directory                                                                                                                                                                                                                                                                                                                                            |
|`-f` |`--offset`             |`0`                               |start frame offset                                                                                                                                                                                                                                                                                                                                                             |
|`-n` |`--nframes`            |`2304`                            |number of frames to process                                                                                                                                                                                                                                                                                                                                                    |
|`-c` |`--channels`           |`['DAPI', 'TRITC', 'CY5', 'FITC']`|channel names                                                                                                                                                                                                                                                                                                                                                                  |
|`-s` |`--starts`             |`[1, 2305, 4609, 9217]`           |channel start indices                                                                                                                                                                                                                                                                                                                                                          |
|`-F` |`--format`             |`Tile%06d.tif`                    |image name format                                                                                                                                                                                                                                                                                                                                                              |
|`-t` |`--threads`            |`0`                               |number of threads for parallel processing                                                                                                                                                                                                                                                                                                                                      |
|`-T` |`--target_channel`     |`TRITC`                           |target channel name for LEV detection                                                                                                                                                                                                                                                                                                                                          |
|`-L` |`--low`                |`99.9`                            |low threshold for target channel segmentation [percentile]                                                                                                                                                                                                                                                                                                                     |
|`-H` |`--high`               |`2`                               |high threshold for target channel segmentation [ratio-to-median]                                                                                                                                                                                                                                                                                                               |
|`-k` |`--kernel`             |`45`                              |size of tophat filter kernel                                                                                                                                                                                                                                                                                                                                                   |
|     |`--max_val`            |`65535`                           |maximum pixel value for foreground during thresholding                                                                                                                                                                                                                                                                                                                         |
|`-v` |`--verbose`            |`0`                               |verbosity level                                                                                                                                                                                                                                                                                                                                                                |
|     |`--include_edge_frames`|                                  |include frames that are on the edge of slide                                                                                                                                                                                                                                                                                                                                   |
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
path to slide image masks directory

#### `-f`, `--offset` (Default: 0)
start frame offset

#### `-n`, `--nframes` (Default: 2304)
number of frames to process

#### `-c`, `--channels` (Default: ['DAPI', 'TRITC', 'CY5', 'FITC'])
channel names

#### `-s`, `--starts` (Default: [1, 2305, 4609, 9217])
channel start indices

#### `-F`, `--format` (Default: Tile%06d.tif)
image name format

#### `-t`, `--threads` (Default: 0)
number of threads for parallel processing

#### `-T`, `--target_channel` (Default: TRITC)
target channel name for LEV detection

#### `-L`, `--low` (Default: 99.9)
low threshold for target channel segmentation [percentile]

#### `-H`, `--high` (Default: 2)
high threshold for target channel segmentation [ratio-to-median]

#### `-k`, `--kernel` (Default: 45)
size of tophat filter kernel

#### `--max_val` (Default: 65535)
maximum pixel value for foreground during thresholding

#### `-v`, `--verbose` (Default: 0)
verbosity level

#### `--include_edge_frames`
include frames that are on the edge of slide

#### `--filter` (Default: [])
                 feature range for filtering detected events.
Usage:    <command> --feature_range <feature> <min> <max>
Example:        <command> --feature_range DAPI_mean 0 10000
Acceptable thresholds are listed in the following table:
feature          minimum         maximum                 -------
-------         -------                 area                    0
+inf                 eccentricity    0                  1
<channel>_mean  0                  <MAX_VAL>


