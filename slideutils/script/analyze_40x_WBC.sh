#!/bin/bash

# Analyzing 40X images of WBCs with DAPI + CD45 stains (two channels)

program_path=/home/amin/Dropbox/Education/PhD/CSI/Projects/if_utils/src

# segment images
python $program_path/detect_events.py -i /media/veracrypt1/0B11619/image -o /media/veracrypt1/0B11619/features.txt -m /media/veracrypt1/0B11619/mask -n 50 -c DAPI CY5 -s 1 4609 --thresh_size 249 --thresh_offsets 250 10 --min_seed_dist 15 --exclude_border --include_edge_frames -v;

# extract event images
python $program_path/extract_event_images.py -i /media/veracrypt1/0B11619/image -d /media/veracrypt1/0B11619/features.txt -m /media/veracrypt1/0B11619/mask -o /media/veracrypt1/0B11619/image_data.hdf5 -w 125 -c DAPI CY5 -s 1 4609 -v

# create gallery of identified events
python $program_path/create_gallery.py -i /media/veracrypt1/0B11619/image_data.hdf5 -o $program_path/$1gallery.tif -w 95 -x 30 -y 20 -m 2 -B DAPI -G CY5 -v

# calculate event histograms
python extract_event_hist.py -i /media/veracrypt1/0B11619/image -m /media/veracrypt1/0B11619/mask -o ./../0B11619_ -b 256 -r 0 65536 -f 0 -n 50 -c DAPI CY5 -s 1 4609 -t 4 -T CY5 --include_edge_frames -v
python extract_event_hist.py -i /media/veracrypt1/0B11619/image -m /media/veracrypt1/0B11619/mask -o ./../0B11619_ -b 256 -r 0 65536 -f 0 -n 50 -c DAPI CY5 -s 1 4609 -t 4 -T DAPI --include_edge_frames -v
