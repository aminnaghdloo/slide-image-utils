#!/home/amin/miniconda3/envs/nn/bin/python
from if_utils.script.if_utils import *
import h5py

# Inputs
data_path = sys.argv[1]
mask_dir = sys.argv[2]
slide_id = sys.argv[3]
verbose = False

print(f"Input slide_id: {slide_id}")
os.makedirs(f"{data_path}/{slide_id}/{mask_dir}", exist_ok = True)

width=35
mask_flag=False
q1 = 99.9
median_coeff = 2
channel_id = 1


# Parameters
filterSize =(45, 45)
max_val = 65535
tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, filterSize)

features = []
images = []

for frame_id in range(1, n_frames+1):
    
    # Skipping edge frames
    if frame_id <= 48 or frame_id > 2256 or frame_id % 24 in [0, 1, 2, 23]:
        continue

    ev_mask_name = f"{data_path}/{slide_id}/{mask_dir}/{frame_id}.tif"

    # Reading the image
    image = getFrameImage(slide_id, frame_id)
    
    # Preprocessing image for masking 
    for i in range(image.shape[-1]):
        image[..., i] = cv2.morphologyEx(image[..., i], cv2.MORPH_TOPHAT,
                                         tophat_kernel)
    tritc_image = image[..., channel_id]
    tritc_blur = tritc_image

    # Thresholding foreground and seeds
    th1 = np.percentile(tritc_blur, q1)
    ret, foreground = cv2.threshold(tritc_blur, th1, max_val, cv2.THRESH_BINARY)
    masked_image = tritc_blur[foreground != 0]
    th2 = median_coeff * np.median(masked_image)
    ret, seeds = cv2.threshold(tritc_blur, th2, max_val, cv2.THRESH_BINARY)
    seeds = measure.label(seeds)
    
    if seeds.max() != 0:
        mask = segmentation.watershed(-foreground, seeds, mask=foreground)
        props = extractMaskInfo(image, mask, slide_id, frame_id)
        crops = extractEventCrops(image, mask, props, width, mask_flag)
        features.append(props)
        images.append(crops)
        cv2.imwrite(ev_mask_name, mask.astype('uint16'))
    else:
        cv2.imwrite(ev_mask_name, np.zeros_like(mask))
    
    if verbose:
        print(f"Frame:\t{frame_id},\tCK events:\t{seeds.max()}")
    
    
    
images = np.concatenate(images, axis=0)
features = pd.concat(features, ignore_index=True)

# Filtering based on size and intensity
thresh = {
        'area': [15, 5000],
        'eccen': [0, 1],
        'dapi': [0, 10000],
        'tritc': [0, 65535],
        'cy5': [0, 65535],
        'fitc': [0, 65535]
    }

sel = (features.area > thresh['area'][0]) &\
      (features.area <= thresh['area'][1]) &\
      (features.eccentricity > thresh['eccen'][0]) &\
      (features.eccentricity <= thresh['eccen'][1]) &\
      (features.DAPI_mean > thresh['dapi'][0]) &\
      (features.DAPI_mean <= thresh['dapi'][1]) &\
      (features.TRITC_mean > thresh['tritc'][0]) &\
      (features.TRITC_mean <= thresh['tritc'][1]) &\
      (features.CY5_mean > thresh['cy5'][0]) &\
      (features.CY5_mean <= thresh['cy5'][1]) &\
      (features.FITC_mean > thresh['fitc'][0]) &\
      (features.FITC_mean <= thresh['fitc'][1])

images = images[sel]
features = features[sel]

with h5py.File(f"{data_path}/{slide_id}/{slide_id}_LEVs.hdf5", "w") as hf:
    hf.create_dataset('images', data=images)

features.to_hdf(f"{data_path}/{slide_id}/{slide_id}_LEVs.hdf5", key='features')

print(f"Successful EV extraction from {slide_id}! (total: {len(images)})")
