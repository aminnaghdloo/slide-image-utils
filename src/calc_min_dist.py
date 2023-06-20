# Description: This script calculates the minimum distance between 
# each LEV and the two closest WBCs.
from scipy.spatial import KDTree
import numpy as np
import pandas as pd
import sys

def main():

    # read the LEV file
    LEV_file_name = sys.argv[1]
    WBC_dir_name = sys.argv[2]

    data_types = {'slide_id': str, 'frame_id': int}
    LEV_df = pd.read_table(LEV_file_name, sep='\t', dtype=data_types)

    dist_dict = {'r':[],
                 'cell_idw':[], 'xw':[], 'yw':[], 'rw':[],
                 'min_dist':[], 'gap':[],
                 'cell_idw2':[], 'xw2':[], 'yw2':[], 'rw2':[],
                 'min_dist2':[], 'gap2':[],
                 'gap_c':[], 'min_dist_c':[], 'dapi_count':[]}
    
    for index, row in LEV_df.iterrows():

        # read the WBC file
        WBC_df = pd.read_table(
            f"{WBC_dir_name}/{row['slide_id']}_{row['frame_id']}.txt")
        dapi_count = len(WBC_df)
        
        # if there are no WBCs or only one, create a dummy dataframe
        if len(WBC_df) < 2:
            WBC_df = pd.DataFrame({'cell_id': [0, 0], 'area': [1, 1],
                                   'x': [10000, 10000], 'y': [10000, 10000]})

        # get the coordinates of the LEV and WBCs
        WBC_coords = list(zip(WBC_df['x'], WBC_df['y']))
        LEV_coord = (row['x'], row['y'])
        
        # get the two closest WBCs
        tree = KDTree(WBC_coords)
        distances, indices = tree.query(LEV_coord, k=2)
        
        # save the distances and coordinates
        r = (row['area'] / np.pi) ** 0.5
        rw = (WBC_df.iloc[indices[0]]['area'] / np.pi) ** 0.5
        rw2 = (WBC_df.iloc[indices[1]]['area'] / np.pi) ** 0.5
        gap = distances[0] - r - rw
        gap2 = distances[1] - r - rw2

        if -gap > r + rw - abs(r - rw) and -gap < r + rw:
            gap_c = gap2
            min_dist_c = distances[1]
        else:
            gap_c = gap
            min_dist_c = distances[0]

        dist_dict['r'].append(r)
        dist_dict['cell_idw'].append(WBC_df.iloc[indices[0]]['cell_id'])
        dist_dict['xw'].append(WBC_df.iloc[indices[0]]['x'])
        dist_dict['yw'].append(WBC_df.iloc[indices[0]]['y'])
        dist_dict['rw'].append(rw)
        dist_dict['min_dist'].append(distances[0])
        dist_dict['gap'].append(gap)
        dist_dict['cell_idw2'].append(WBC_df.iloc[indices[1]]['cell_id'])
        dist_dict['xw2'].append(WBC_df.iloc[indices[1]]['x'])
        dist_dict['yw2'].append(WBC_df.iloc[indices[1]]['y'])
        dist_dict['rw2'].append(rw2)
        dist_dict['min_dist2'].append(distances[1])
        dist_dict['gap2'].append(gap2)
        dist_dict['gap_c'].append(gap_c)
        dist_dict['min_dist_c'].append(min_dist_c)
        dist_dict['dapi_count'].append(dapi_count)

        print(f"processed {index} / {len(LEV_df)} against {len(WBC_df)} WBCs")

    dist_df = pd.DataFrame(dist_dict)
    out_df = pd.concat([LEV_df.iloc[:, :6], dist_df], axis=1)
    out_df.to_csv(f"{LEV_file_name[:-4]}_min_dist.txt", sep='\t', index=False)
    print(f"Minimum distances were calculated for {LEV_file_name}")


if __name__ == '__main__':
    main()
