import numpy as np

from rasterio.features import geometry_mask
from shapely.geometry import mapping

def crop_image_by_mask(image, mask, channel_first=True):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # Get the min/max row/column indices
    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    # Add 1 to max_row and max_col to make slicing inclusive
    max_row += 1
    max_col += 1

    # Crop the image based on the bounding box
    if image.ndim == 2: # (H, W)
        cropped_image = image[min_row:max_row, min_col:max_col]
    elif image.ndim == 3:
        if channel_first: # (C, H, W)
            cropped_image = image[:, min_row:max_row, min_col:max_col]
        else: # (H, W, C)
            cropped_image = image[min_row:max_row, min_col:max_col, :]
    else:
        cropped_image = image

    return cropped_image


def generate_mask_from_polygon(polygon, img_transform, img_x, img_y):
    mask = geometry_mask([mapping(polygon)],
                         transform=img_transform,
                         invert=True,
                         out_shape=(img_x, img_y))

    return mask


def apply_mask_to_image(image, mask, fill_value=0):
    """
    Applies a boolean mask to an image. Pixels where the mask is False
    will be set to the fill_value.

    This function handles:
    - Grayscale images (H, W)
    - Multi-band images (C, H, W) like (1, 4000, 4000) or (7, 4000, 4000)
    - RGB/RGBA images (H, W, C)

    Args:
        image (np.ndarray): The input image.
                            Expected shapes: (H, W), (C, H, W), or (H, W, C).
        mask (np.ndarray): A boolean mask array (True for areas to keep,
                           False for areas to mask out). Must have the
                           same spatial dimensions (H, W) as the image.
        fill_value (int/float): The value to fill masked-out areas with.
                                Defaults to 0 (black).

    Returns:
        np.ndarray: The masked image.
    """
    # Create a copy to avoid modifying the original image
    masked_image = image.copy()

    # Determine spatial dimensions (Height, Width) and channel position
    if image.ndim == 2:  # (H, W) - Grayscale
        image_spatial_shape = image.shape
        channel_first = False
    elif image.ndim == 3:
        # Heuristic to determine if channels are first or last
        # If first dimension is much smaller than others, assume (C, H, W)
        if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
            image_spatial_shape = image.shape[1:]  # (H, W)
            channel_first = True
        else:  # Assume (H, W, C)
            image_spatial_shape = image.shape[:2]  # (H, W)
            channel_first = False
    else:
        raise ValueError("Unsupported image dimensions. Expected 2 or 3 dimensions.")

    # Validate mask shape against image spatial shape
    if mask.shape != image_spatial_shape:
        raise ValueError(
            f"Mask spatial dimensions {mask.shape} do not match image spatial dimensions {image_spatial_shape}.")

    # Apply the mask
    if image.ndim == 2:  # Grayscale (H, W)
        masked_image[~mask] = fill_value
    elif image.ndim == 3:
        if channel_first:  # (C, H, W)
            # Apply the 2D mask to each channel
            for i in range(masked_image.shape[0]):
                masked_image[i, ~mask] = fill_value
        else:  # (H, W, C)
            # Expand mask to match channels for broadcasting
            expanded_mask = mask[:, :, np.newaxis]  # Reshape mask to (H, W, 1)
            masked_image[~expanded_mask] = fill_value

    return masked_image

def read_geotiff(file_name):
    """
    Reads a GeoTIFF file and extracts its content including the image data,
    affine transform, NoData value, and coordinate reference system (CRS).

    :param file_name: Path to the GeoTIFF file to be read.
    :type file_name: str
    :return: A tuple containing the following elements:
        - img: The image data as a NumPy array of float32.
        - transform: Affine transformation representing the spatial relationship
          of pixels.
        - nodata: Value used to represent "NoData" in the raster, if specified.
        - tif_crs: Coordinate Reference System (CRS) of the GeoTIFF.
    :rtype: tuple
    """
    with rasterio.open(file_name) as s:
        img = s.read().astype(np.float32)
        transform = s.transform
        nodata = s.nodata
        tif_crs = s.crs

    return img, transform, nodata, tif_crs