## siu_filter_hdf
### Usage
```
usage: argdown [-h] -i INPUT -o OUTPUT [--filter FILTER FILTER FILTER]
```
### Arguments
#### Quick reference table
|Short|Long      |Default|Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|-----|----------|-------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`-h` |`--help`  |       |show this help message and exit                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|`-i` |`--input` |`None` |Input HDF5 files                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|`-o` |`--output`|`None` |Output HDF5 file                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|     |`--filter`|`[]`   |
        feature range for filtering detected events.

        Usage:      <command> --feature_range <feature> <min> <max>
        Example:    <command> --feature_range DAPI_mean 0 10000

        Acceptable thresholds are listed in the following table:

        feature         minimum     maximum
        -------         -------     -------
        area            0           +inf
        eccentricity    0           1
        <channel>_mean  0           <MAX_VAL>
        |

#### `-h`, `--help`
show this help message and exit

#### `-i`, `--input` (Default: None)
Input HDF5 files

#### `-o`, `--output` (Default: None)
Output HDF5 file

#### `--filter` (Default: [])
         feature range for filtering detected events.          Usage:
<command> --feature_range <feature> <min> <max>         Example:    <command>
--feature_range DAPI_mean 0 10000          Acceptable thresholds are listed in
the following table:          feature         minimum     maximum
-------         -------     -------         area            0           +inf
eccentricity    0           1         <channel>_mean  0           <MAX_VAL>


