## siu_create_gallery
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

#### `-h`, `--help`
show this help message and exit

#### `-i`, `--input` (Default: None)
path to HDF file containing event images

#### `-d`, `--data` (Default: None)
path to tab-delimited events data file

#### `-o`, `--output` (Default: None)
path to output gallery image

#### `-w`, `--width` (Default: 35)
size of each event image crop (odd)

#### `-x`, `--nx` (Default: 15)
number of images along x axis in gallery

#### `-y`, `--ny` (Default: 15)
number of images along y axis in gallery

#### `-m`, `--mask_flag` (Default: None)
mask flag

#### `-R`, `--red` (Default: [])
channel(s) to be shown in red color

#### `-G`, `--green` (Default: [])
channel(s) to be shown in green color

#### `-B`, `--blue` (Default: [])
channel(s) to be shown in blue color

#### `-g`, `--gain` (Default: [1, 1, 1, 1])
gains applied to each channel

#### `-v`, `--verbose` (Default: 0)
verbosity level

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


