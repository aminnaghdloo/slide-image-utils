## siu_extract_event_images
### Usage
```
siu_extract_event_images [-h] -i IMAGE -d DATA -o OUTPUT [-w WIDTH] [-m MASK]
                         [-c CHANNELS [CHANNELS ...]] [-s STARTS [STARTS ...]]
                         [-F FORMAT] [-v] [--filter FILTER FILTER FILTER]
```
### Arguments
#### Quick reference table
|Short|Long        |Default                           |Description                                                                                                                                                                                                                                                                                                                                                              |
|-----|------------|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`-h` |`--help`    |                                  |show this help message and exit                                                                                                                                                                                                                                                                                                                                          |
|`-i` |`--image`   |`None`                            |path to slide images directory                                                                                                                                                                                                                                                                                                                                           |
|`-d` |`--data`    |`None`                            |path to tab-delimited event data file with <frame_id>	<x>	<y>                                                                                                                                                                                                                                                                                                            |
|`-o` |`--output`  |`None`                            |path to output hdf5 file including event both images and data                                                                                                                                                                                                                                                                                                            |
|`-w` |`--width`   |`35`                              |size of the event images to be cropped from slide images (odd)                                                                                                                                                                                                                                                                                                           |
|`-m` |`--mask`    |`None`                            |path to mask directory                                                                                                                                                                                                                                                                                                                                                   |
|`-c` |`--channels`|`['DAPI', 'TRITC', 'CY5', 'FITC']`|channel names                                                                                                                                                                                                                                                                                                                                                            |
|`-s` |`--starts`  |`[1, 2305, 4609, 9217]`           |channel start indices                                                                                                                                                                                                                                                                                                                                                    |
|`-F` |`--format`  |`Tile%06d.tif`                    |image name format                                                                                                                                                                                                                                                                                                                                                        |
|`-v` |`--verbose` |`0`                               |verbosity level                                                                                                                                                                                                                                                                                                                                                          |
|     |`--filter`  |`[]`                              |
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


