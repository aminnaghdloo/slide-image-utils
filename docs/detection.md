# Detection Tools

## Detect Cells

[Add description here]

### Usage
```
siu_detect_cells [-h] -i INPUT -o OUTPUT [-m MASK_PATH] [-f OFFSET] [-n NFRAMES]
                 [-c CHANNELS [CHANNELS ...]] [-s STARTS [STARTS ...]]
                 [-F FORMAT [FORMAT ...]] [-v] [-t THREADS]
                 [--tophat_size TOPHAT_SIZE] [--open_size OPEN_SIZE]
                 [--blur_size BLUR_SIZE] [--blur_sigma BLUR_SIGMA]
                 [--thresh_size THRESH_SIZE]
                 [--thresh_offsets THRESH_OFFSETS [THRESH_OFFSETS ...]]
                 [--min_seed_dist MIN_SEED_DIST]
                 [--seed_channel SEED_CHANNEL [SEED_CHANNEL ...]]
                 [--mask_channels MASK_CHANNELS [MASK_CHANNELS ...]]
                 [--exclude_border] [--include_edge_frames]
                 [--selected_frames [SELECTED_FRAMES ...]] [--extract_images]
                 [-w WIDTH] [--mask_flag] [--sort SORT SORT]
                 [--filter FILTER FILTER FILTER]
```
### Arguments
|Short|Long                   |Default                           |Description                                                                                              |
|-----|-----------------------|----------------------------------|---------------------------------------------------------------------------------------------------------|
|`-h` |`--help`               |                                  |show this help message and exit                                                                          |
|`-i` |`--input`              |`None`                            |input path to slide images                                                                               |
|`-o` |`--output`             |`None`                            |output path                                                                                              |
|`-m` |`--mask_path`          |`None`                            |mask path to save frame masks if needed                                                                  |
|`-f` |`--offset`             |`0`                               |start frame offset                                                                                       |
|`-n` |`--nframes`            |`2304`                            |number of frames                                                                                         |
|`-c` |`--channels`           |`['DAPI', 'TRITC', 'CY5', 'FITC']`|channel names                                                                                            |
|`-s` |`--starts`             |`[1, 2305, 4609, 9217]`           |channel start indices                                                                                    |
|`-F` |`--format`             |`['Tile%06d.tif']`                |image name format                                                                                        |
|`-v` |`--verbose`            |`0`                               |verbosity level                                                                                          |
|`-t` |`--threads`            |`0`                               |number of threads for parallel processing                                                                |
|     |`--tophat_size`        |`45`                              |TopHat filter kernel size                                                                                |
|     |`--open_size`          |`5`                               |Open morphological filter kernel size                                                                    |
|     |`--blur_size`          |`5`                               |Gaussian blur filter kernel size                                                                         |
|     |`--blur_sigma`         |`2`                               |Gaussian blur filter parameter sigma                                                                     |
|     |`--thresh_size`        |`25`                              |adaptive threshold window size                                                                           |
|     |`--thresh_offsets`     |`[-2000, -2000, -2000, -2000]`    |adaptive threshold offset values for channels (negative)                                                 |
|     |`--min_seed_dist`      |`7`                               |minimum allowed distance [pixels] between each pair of seeds                                             |
|     |`--seed_channel`       |`['DAPI']`                        |channel name(s) to use as seed                                                                           |
|     |`--mask_channels`      |`['DAPI', 'TRITC', 'CY5', 'FITC']`|channels to segment                                                                                      |
|     |`--exclude_border`     |                                  |exclude events that are on image borders                                                                 |
|     |`--include_edge_frames`|                                  |include frames that are on the edge of slide                                                             |
|     |`--selected_frames`    |`[]`                              |list of selected frames to be processed                                                                  |
|     |`--extract_images`     |                                  |extract images of detected events and output hdf5 file                                                   |
|`-w` |`--width`              |`35`                              |size of the event images to be cropped from slide images (odd). Works only when --extract_images is set. |
|     |`--mask_flag`          |                                  |store event masks when extracting images                                                                 |
|     |`--sort`               |`[]`                              |sort events based on feature values. [Explained here]                                                    |
|     |`--filter`             |`[]`                              |feature range for filtering detected events. [Explained here]                                            |

`--sort`

                 sort events based on feature values.                  Usage:
<command> --sort <feature> <order>                 Example:        <command>
--sort TRITC_mean I                 order:    I: Increasing / D: Decreasing

`--filter`

                 feature range for filtering detected events.
Usage:    <command> --feature_range <feature> <min> <max>
Example:        <command> --feature_range DAPI_mean 0 10000

### Example

## Detect PACCs

[Add description here]

### Usage
```
siu_detect_PACCs [-h] -i INPUT -o OUTPUT [-m MASK_PATH] [-f OFFSET] [-n NFRAMES]
                 [-c CHANNELS [CHANNELS ...]] [-s STARTS [STARTS ...]]
                 [-F FORMAT [FORMAT ...]] [-v] [-t THREADS]
                 [--tophat_size TOPHAT_SIZE] [--open_size OPEN_SIZE]
                 [--blur_size BLUR_SIZE] [--blur_sigma BLUR_SIGMA]
                 [--thresh_size THRESH_SIZE]
                 [--thresh_offsets THRESH_OFFSETS [THRESH_OFFSETS ...]]
                 [--min_seed_dist MIN_SEED_DIST]
                 [--seed_channel SEED_CHANNEL [SEED_CHANNEL ...]]
                 [--mask_channels MASK_CHANNELS [MASK_CHANNELS ...]]
                 [--exclude_border] [--include_edge_frames]
                 [--selected_frames [SELECTED_FRAMES ...]] [--extract_images]
                 [-w WIDTH] [--mask_flag] [--sort SORT SORT]
                 [--filter FILTER FILTER FILTER]
```
### Arguments
|Short|Long                   |Default                           |Description                                                                                              |
|-----|-----------------------|----------------------------------|---------------------------------------------------------------------------------------------------------|
|`-h` |`--help`               |                                  |show this help message and exit                                                                          |
|`-i` |`--input`              |`None`                            |input path to slide images                                                                               |
|`-o` |`--output`             |`None`                            |output path                                                                                              |
|`-m` |`--mask_path`          |`None`                            |mask path to save frame masks if needed                                                                  |
|`-f` |`--offset`             |`0`                               |start frame offset                                                                                       |
|`-n` |`--nframes`            |`2304`                            |number of frames                                                                                         |
|`-c` |`--channels`           |`['DAPI', 'TRITC', 'CY5', 'FITC']`|channel names                                                                                            |
|`-s` |`--starts`             |`[1, 2305, 4609, 9217]`           |channel start indices                                                                                    |
|`-F` |`--format`             |`'Tile%06d.tif'`                  |image name format                                                                                        |
|`-v` |`--verbose`            |`0`                               |verbosity level                                                                                          |
|`-t` |`--threads`            |`0`                               |number of threads for parallel processing                                                                |
|     |`--tophat_size`        |`45`                              |TopHat filter kernel size                                                                                |
|     |`--open_size`          |`5`                               |Open morphological filter kernel size                                                                    |
|     |`--blur_size`          |`5`                               |Gaussian blur filter kernel size                                                                         |
|     |`--blur_sigma`         |`2`                               |Gaussian blur filter parameter sigma                                                                     |
|     |`--thresh_size`        |`25`                              |adaptive threshold window size                                                                           |
|     |`--thresh_offsets`     |`[-2000, -2000, -2000, -2000]`    |adaptive threshold offset values for channels (negative)                                                 |
|     |`--min_seed_dist`      |`7`                               |minimum allowed distance [pixels] between each pair of seeds                                             |
|     |`--nucleus_channel`    |`'DAPI'`                          |channel name(s) to use as seed                                                                           |
|     |`--mask_channels`      |`['DAPI', 'TRITC', 'CY5', 'FITC']`|channels to segment                                                                                      |
|     |`--exclude_border`     |                                  |exclude events that are on image borders                                                                 |
|     |`--include_edge_frames`|                                  |include frames that are on the edge of slide                                                             |
|     |`--selected_frames`    |`[]`                              |list of selected frames to be processed                                                                  |
|     |`--extract_images`     |                                  |extract images of detected events and output hdf5 file                                                   |
|`-w` |`--width`              |`35`                              |size of the event images to be cropped from slide images (odd). Works only when --extract_images is set. |
|     |`--mask_flag`          |                                  |store event masks when extracting images                                                                 |
|     |`--sort`               |`[]`                              |sort events based on feature values. [Explained here]                                                    |
|     |`--filter`             |`[]`                              |feature range for filtering detected events. [Explained here]                                            |

### Example

[Add an example here]

## Detect LEVs

[Add description here]

### Usage
```
siu_detect_LEVs [-h] -i INPUT -o OUTPUT [-m MASK] [-f OFFSET] [-n NFRAMES]
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
|Short|Long                   |Default                           |Description                                                                                              |
|-----|-----------------------|----------------------------------|---------------------------------------------------------------------------------------------------------|
|`-h` |`--help`               |                                  |show this help message and exit                                                                          |
|`-i` |`--input`              |`None`                            |input path to slide images                                                                               |
|`-o` |`--output`             |`None`                            |output path                                                                                              |
|`-m` |`--mask_path`          |`None`                            |mask path to save frame masks if needed                                                                  |
|`-f` |`--offset`             |`0`                               |start frame offset                                                                                       |
|`-n` |`--nframes`            |`2304`                            |number of frames                                                                                         |
|`-c` |`--channels`           |`['DAPI', 'TRITC', 'CY5', 'FITC']`|channel names                                                                                            |
|`-s` |`--starts`             |`[1, 2305, 4609, 9217]`           |channel start indices                                                                                    |
|`-F` |`--format`             |`'Tile%06d.tif'`                  |image name format                                                                                        |
|`-C` |`--bg_cell_channels`   |`['DAPI', 'CY5']`                 |name of channels for which background cells are positive                                                 |
|`-t` |`--threads`            |`0`                               |number of threads for parallel processing                                                                |
|`-T` |`--target_channel`     |`TRITC`                           |target channel name for LEV                                                                              |
|`-L` |`--low`                |`99.7`                            |low threshold for segmentation [percentile]                                                              |
|`-H` |`--high`               |`2`                               |high threshold for segmentation [ratio-to-median]                                                        |
|`-k` |`--kernel`             |`75`                              |size of tophat filter kernel                                                                             |
|     |`--max_val`            |`65535`                           |maximum pixel value for foreground during thresholding                                                   |
|`-v` |`--verbose`            |`0`                               |verbosity level                                                                                          |
|     |`--exclude_border`     |                                  |exclude events that are on image borders                                                                 |
|     |`--include_edge_frames`|                                  |include frames that are on the edge of slide                                                             |
|     |`--selected_frames`    |`[]`                              |list of selected frames to be processed                                                                  |
|     |`--extract_images`     |                                  |extract images of detected events and output hdf5 file                                                   |
|`-w` |`--width`              |`35`                              |size of the event images to be cropped from slide images (odd). Works only when --extract_images is set. |
|     |`--mask_flag`          |                                  |store event masks when extracting images                                                                 |
|     |`--sort`               |`[]`                              |sort events based on feature values. [Explained here]                                                    |
|     |`--filter`             |`[]`                              |feature range for filtering detected events. [Explained here]                                            |

### Example

[Add an example here]