import numpy as np
import argparse
import pandas as pd
import h5py

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge HDF5 files')
    parser.add_argument('-i', '--input', type=str, nargs='+',
                        help='Input HDF5 files')
    parser.add_argument('-o', '--output', type=str, help='Output HDF5 file')
    parser.add_argument('-v', '--verbose', action='store_true',help='Verbose mode')

    args = parser.parse_args()

    # Read in the data
    all_images = []
    all_features = []
    all_masks = []
    all_channels = []

    for input_file in args.input:

        slide_id = input_file.split('/')[-1].split('.')[0]

        if args.verbose:
            print(f'Reading {input_file}...')

        with h5py.File(input_file, 'r') as f:
            all_images.append(f['images'][:])
            # all_masks.append(f['masks'][:])
            all_channels.append(f['channels'][:])
            # channels.append(f.attrs['channels'])
        
        df = pd.read_hdf(input_file, mode='r', key='features')
        #df.insert(0, 'slide_id', slide_id)
        all_features.append(df)

    # Concatenate the data
    images = np.concatenate(all_images, axis=0)
    # masks = np.concatenate(all_masks, axis=0)
    channels = all_channels[0]
    features = pd.concat(all_features, axis=0)

    # Write out the data
    if args.verbose:
        print(f'Writing {args.output}...')
        
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('images', data=images)
        # f.create_dataset('masks', data=masks)
        f.create_dataset('channels', data=channels)
    
    features.reset_index(drop=True, inplace=True)
    features.to_hdf(args.output, key='features', mode='a')

    print('Done!')
