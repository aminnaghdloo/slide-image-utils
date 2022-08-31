from script.if_utils.script.if_utils import *

frame_count = 2304
slide_id = sys.argv[1]

count_path = "data/NEvL/counts"

dapi_count = []

for i in range(10):
    frame = Frame(slide_id=slide_id, frame_id=i+1)
    dapi_count.append([i + 1, frame.count])
    
dapi_count = pd.DataFrame(dapi_count, columns=['frame_id', 'cell_count'])
dapi_count.to_csv(f"{count_path}/{slide_id}_dapicount.txt", sep='\t',
                  index=False)