## siu_extract_closest_events
### Usage
```
usage: argdown [-h] -r REFERENCE -q QUERY -o OUTPUT
               [--matching_columns [MATCHING_COLUMNS ...]]
```
### Arguments
#### Quick reference table
|Short|Long                |Default       |Description                                                                    |
|-----|--------------------|--------------|-------------------------------------------------------------------------------|
|`-h` |`--help`            |              |show this help message and exit                                                |
|`-r` |`--reference`       |`None`        |Path to input tab-delimited text data of reference events                      |
|`-q` |`--query`           |`None`        |Path to input tab-delimited text data of query events                          |
|`-o` |`--output`          |`None`        |Path to output data                                                            |
|     |`--matching_columns`|`['frame_id']`|List of column names on which events match across reference 
				and query data|

#### `-h`, `--help`
show this help message and exit

#### `-r`, `--reference` (Default: None)
Path to input tab-delimited text data of reference events

#### `-q`, `--query` (Default: None)
Path to input tab-delimited text data of query events

#### `-o`, `--output` (Default: None)
Path to output data

#### `--matching_columns` (Default: ['frame_id'])
List of column names on which events match across reference
and query data


