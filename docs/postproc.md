# Postprocessing Tools

## Extract Event Images

[Add description here]

### Usage
```
siu_extract_event_images [-h] -i IMAGE -d DATA -o OUTPUT [-w WIDTH] [-m MASK]
                         [-c CHANNELS [CHANNELS ...]] [-s STARTS [STARTS ...]]
                         [-F FORMAT] [-v] [--filter FILTER FILTER FILTER]
```
### Arguments
|Short|Long                   |Default                           |Description                                                                                              |
|-----|-----------------------|----------------------------------|---------------------------------------------------------------------------------------------------------|
|`-h` |`--help`               |                                  |show this help message and exit                                                                          |
|`-i` |`--image`              |`None`                            |path to slide images directory                                                                           |
|`-d` |`--data`               |`None`                            |path to tab-delimited event data file with columns <frame_id> <x> <y>                                    |
|`-o` |`--output`             |`None`                            |path to output hdf5 file including event both images and data                                            |
|`-w` |`--width`              |`35`                              |size of the event images to be cropped from slide images (odd)                                           |
|`-m` |`--mask`               |`None`                            |path to mask directory                                                                                   |
|`-c` |`--channels`           |`['DAPI', 'TRITC', 'CY5', 'FITC']`|channel names                                                                                            |
|`-s` |`--starts`             |`[1, 2305, 4609, 9217]`           |channel start indices                                                                                    |
|`-F` |`--format`             |`Tile%06d.tif`                    |image name format                                                                                        |
|`-v` |`--verbose`            |`0`                               |verbosity level                                                                                          |
|     |`--filter`             |`[]`                              |feature range for filtering detected events. [Explained here]                                            |

### Example

[Add an example here]

## Create Gallery

[Add description here]

### Usage
```
siu_create_gallery [-h] -i INPUT [-d DATA] -o OUTPUT [-w WIDTH] [-x NX] [-y NY]
                   [-m {crop,overlay}] -R RED [RED ...] -G GREEN [GREEN ...] -B
                   BLUE [BLUE ...] [-g GAIN [GAIN ...]] [-v] [--sort SORT SORT]
                   [--filter FILTER FILTER FILTER]
```
### Arguments
|Short|Long                   |Default                           |Description                                                                                              |
|-----|-----------------------|----------------------------------|---------------------------------------------------------------------------------------------------------|
|`-h` |`--help`     |              |show this help message and exit                                                                                                        |
|`-i` |`--input`    |`None`        |path to HDF file containing event images                                                                                               |
|`-d` |`--data`     |`None`        |path to tab-delimited events data file                                                                                                 |
|`-o` |`--output`   |`None`        |path to output gallery image                                                                                                           |
|`-w` |`--width`    |`35`          |size of each event image crop (odd)                                                                                                    |
|`-x` |`--nx`       |`15`          |number of images along x axis in gallery                                                                                               |
|`-y` |`--ny`       |`15`          |number of images along y axis in gallery                                                                                               |
|`-m` |`--mask_flag`|`None`        |mask flag                                                                                                                              |
|`-R` |`--red`      |`[]`          |channel(s) to be shown in red color                                                                                                    |
|`-G` |`--green`    |`[]`          |channel(s) to be shown in green color                                                                                                  |
|`-B` |`--blue`     |`[]`          |channel(s) to be shown in blue color                                                                                                   |
|`-g` |`--gain`     |`[1, 1, 1, 1]`|gains applied to each channel                                                                                                          |
|`-v` |`--verbose`  |`0`           |verbosity level                                                                                                                        |
|     |`--sort`     |`[]`          |sort events based on feature values.                                                                                                   |
|     |`--filter`   |`[]`          |feature range for filtering detected events.                                                                                           |

### Example

[Add an example here]

## Create Montages

[Add description here]

### Usage
```
siu_create_montages [-h] -i INPUT [-d DATA] -o OUTPUT [-w WIDTH]
                    [-m {crop,overlay}] -R RED [RED ...] -G GREEN [GREEN ...]
                    -B BLUE [BLUE ...] [-g GAIN [GAIN ...]] [-O [ORDER ...]]
                    [-s] [-v] [--sort SORT SORT] [--filter FILTER FILTER FILTER]
```


### Example

[Add an example here]

## Filter Events [HDF5 Files]

[Add description here]







### Example

[Add an example here]

## Sort Events [HDF5 Files]

[Add description here]






### Example

[Add an example here]

## Merge Events [HDF5 Files]

[Add description here]









### Example

[Add an example here]