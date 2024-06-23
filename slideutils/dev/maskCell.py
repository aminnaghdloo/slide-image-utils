from if_utils.script.if_utils import *

# Parameters
tophat_size = (45, 45)  # Tophat filter size
opening_size = (5, 5)  # Morphological opening filter size
blur_size = (5, 5)  # Gaussian blur filter size
blur_sig = 1  # Gaussian blur sigma value
thresh_winsize = 25  # Adaptive thresholding window size
thresh_offset = -2  # Adaptive thresholding offset value
min_dist = 7  # marker detection adjacency limit [px]
size_minthresh = 10  # minimum size threshold for
force_flag = True

channels.popitem()

# Image processing kernels
tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tophat_size)
opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, opening_size)


def maskWBC(slide_id, frame_id, verbose=True):
    # Load frame image
    image = getFrameImage(slide_id, frame_id)
    image = image // 256
    image = image.astype("uint8")

    # Segment DAPI channel
    dapi_image = image[:, :, 0]
    dapi_image = cv2.morphologyEx(dapi_image, cv2.MORPH_TOPHAT, tophat_kernel)
    dapi_blur = cv2.GaussianBlur(dapi_image, blur_size, blur_sig)
    dapi_mask = cv2.adaptiveThreshold(
        dapi_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        thresh_winsize,
        thresh_offset,
    )
    dapi_mask = fillHull(dapi_mask)

    # Segment TRITC channel
    tritc_image = image[:, :, 1]
    tritc_image = cv2.morphologyEx(
        tritc_image, cv2.MORPH_TOPHAT, tophat_kernel
    )
    tritc_blur = cv2.GaussianBlur(tritc_image, blur_size, blur_sig)
    tritc_mask = cv2.adaptiveThreshold(
        tritc_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        thresh_winsize,
        thresh_offset,
    )
    tritc_mask = fillHull(tritc_mask)

    # Segment Cy5 channel
    cy5_image = image[:, :, 2]
    cy5_image = cv2.morphologyEx(cy5_image, cv2.MORPH_TOPHAT, tophat_kernel)
    cy5_blur = cv2.GaussianBlur(cy5_image, blur_size, blur_sig)
    cy5_mask = cv2.adaptiveThreshold(
        cy5_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        thresh_winsize,
        thresh_offset,
    )
    cy5_mask = fillHull(cy5_mask)

    # Combine channel masks
    merged_mask = cv2.bitwise_or(dapi_mask, cy5_mask)
    merged_mask = cv2.bitwise_or(merged_mask, tritc_mask)
    merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_OPEN, opening_kernel)

    # Distance transform
    dapi_opening = cv2.morphologyEx(dapi_mask, cv2.MORPH_OPEN, opening_kernel)
    dapi_dist = cv2.distanceTransform(dapi_opening, cv2.DIST_L2, 3, cv2.CV_32F)
    cell_dist = cv2.distanceTransform(merged_mask, cv2.DIST_L2, 3, cv2.CV_32F)

    # Marker detection
    local_max_coords = feature.peak_local_max(dapi_dist, min_distance=min_dist)
    local_max_mask = np.zeros(dapi_mask.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = measure.label(local_max_mask)

    if verbose:
        print(f"Identified {markers.max()} WBCs from frame {frame_id}")

    # Watershed segmentation
    nuclei_mask = segmentation.watershed(-dapi_dist, markers, mask=dapi_mask)
    cells_mask = segmentation.watershed(-cell_dist, markers, mask=merged_mask)

    return (nuclei_mask, cells_mask)


def main():
    # Read input
    slide_id = sys.argv[1]
    data_path = sys.argv[2]
    out_path = f"{data_path}/{slide_id}"

    print(f"Input slide_id: {slide_id}")
    os.makedirs(f"{out_path}/WBCmask", exist_ok=True)
    os.makedirs(f"{out_path}/nucleusmask", exist_ok=True)

    counts = []
    # Process all frames
    for frame_id in range(1, n_frames + 1):

        # Skipping edge frames
        if frame_id <= 48 or frame_id > 2256 or frame_id % 24 in [0, 1, 2, 23]:
            continue

        nuclei_mask_name = f"{out_path}/nucleusmask/{frame_id}.tif"
        cells_mask_name = f"{out_path}/WBCmask/{frame_id}.tif"

        if os.path.exists(cells_mask_name) and not force_flag:
            print(f"WBC mask exists for {slide_id}:{frame_id}")

        else:
            nuclei_mask, cells_mask = maskWBC(slide_id, frame_id)
            counts.append([frame_id, nuclei_mask.max()])
            cv2.imwrite(nuclei_mask_name, nuclei_mask.astype("uint16"))
            cv2.imwrite(cells_mask_name, cells_mask.astype("uint16"))

    counts = pd.DataFrame(counts, columns=["frame_id", "dapi_count"])
    counts.to_csv(f"{out_path}/counts.txt", sep="\t", index=False)

    print(f"WBCs were masked successfully for slide: {slide_id}")


if __name__ == "__main__":
    main()
