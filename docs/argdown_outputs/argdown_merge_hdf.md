## siu_merge_hdf
### Usage
```
usage: argdown [-h] -i INPUT [INPUT ...] -o OUTPUT [-m]
               [-a ADD_FILENAME_COLUMN] [-v]
```
### Arguments
#### Quick reference table
|Short|Long                   |Default|Description                                  |
|-----|-----------------------|-------|---------------------------------------------|
|`-h` |`--help`               |       |show this help message and exit              |
|`-i` |`--input`              |`None` |Input HDF5 files                             |
|`-o` |`--output`             |`None` |Output HDF5 file                             |
|`-m` |`--mask_flag`          |       |merge the masks as well                      |
|`-a` |`--add_filename_column`|`None` |column name to which the code adds file names|
|`-v` |`--verbose`            |       |Verbose mode                                 |

#### `-h`, `--help`
show this help message and exit

#### `-i`, `--input` (Default: None)
Input HDF5 files

#### `-o`, `--output` (Default: None)
Output HDF5 file

#### `-m`, `--mask_flag`
merge the masks as well

#### `-a`, `--add_filename_column` (Default: None)
column name to which the code adds file names

#### `-v`, `--verbose`
Verbose mode


