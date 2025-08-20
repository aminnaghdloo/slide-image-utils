## siu_create_40x_montages
### Usage
```
usage: argdown [-h] -i INPUT [-d DATA] -o OUTPUT [-w WIDTH]
               [-c CHANNELS [CHANNELS ...]] [-g GAIN [GAIN ...]]
               [-R RED [RED ...]] [-G GREEN [GREEN ...]] [-B BLUE [BLUE ...]]
               [-O ORDER [ORDER ...]] [-s] [-v]
```
### Arguments
#### Quick reference table
|Short|Long        |Default                           |Description                                                                            |
|-----|------------|----------------------------------|---------------------------------------------------------------------------------------|
|`-h` |`--help`    |                                  |show this help message and exit                                                        |
|`-i` |`--input`   |`None`                            |path to 40x image directory containing 40x frame images                                |
|`-d` |`--data`    |`None`                            |path to tab-delimited reimaged data file                                               |
|`-o` |`--output`  |`None`                            |path to output gallery image                                                           |
|`-w` |`--width`   |`35`                              |size of each event image crop (odd)                                                    |
|`-c` |`--channels`|`['DAPI', 'TRITC', 'CY5', 'FITC']`|channel(s) to include while reading images                                             |
|`-g` |`--gain`    |`[1, 1, 1, 1]`                    |gains applied to each channel                                                          |
|`-R` |`--red`     |`['TRITC', 'FITC']`               |channel(s) to be shown in red color                                                    |
|`-G` |`--green`   |`['CY5', 'FITC']`                 |channel(s) to be shown in green color                                                  |
|`-B` |`--blue`    |`['DAPI', 'FITC']`                |channel(s) to be shown in blue color                                                   |
|`-O` |`--order`   |`['DAPI', 'TRITC', 'FITC', 'CY5']`|order of channels in grayscale section of the montage                                  |
|`-s` |`--separate`|                                  |
		save montages individually for each event as 
		<cell_id>-<frame_id>-<x>-<y>_40x.jpg|
|`-v` |`--verbose` |`0`                               |verbosity level                                                                        |

#### `-h`, `--help`
show this help message and exit

#### `-i`, `--input` (Default: None)
path to 40x image directory containing 40x frame images

#### `-d`, `--data` (Default: None)
path to tab-delimited reimaged data file

#### `-o`, `--output` (Default: None)
path to output gallery image

#### `-w`, `--width` (Default: 35)
size of each event image crop (odd)

#### `-c`, `--channels` (Default: ['DAPI', 'TRITC', 'CY5', 'FITC'])
channel(s) to include while reading images

#### `-g`, `--gain` (Default: [1, 1, 1, 1])
gains applied to each channel

#### `-R`, `--red` (Default: ['TRITC', 'FITC'])
channel(s) to be shown in red color

#### `-G`, `--green` (Default: ['CY5', 'FITC'])
channel(s) to be shown in green color

#### `-B`, `--blue` (Default: ['DAPI', 'FITC'])
channel(s) to be shown in blue color

#### `-O`, `--order` (Default: ['DAPI', 'TRITC', 'FITC', 'CY5'])
order of channels in grayscale section of the montage

#### `-s`, `--separate`
                 save montages individually for each event as
<cell_id>-<frame_id>-<x>-<y>_40x.jpg

#### `-v`, `--verbose` (Default: 0)
verbosity level


