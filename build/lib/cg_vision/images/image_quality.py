import cv2
import numpy as np


def calculate_relative_exposure(img: np.ndarray) -> float:
    """
    Calculates the average pixel value of an image as a measure of exposure.

    Args:
      img: Path to the image file.

    Returns:
      float: The average pixel value (0-255) representing the exposure.
    """
    # Calculate the average pixel value across all channels
    average_pixel_value = img.mean()

    relative_exposure = (average_pixel_value / 255) * 100

    return round(relative_exposure, 1)


def count_undersaturated_pixels(img: np.ndarray, threshold: int = 20) -> float:
    """
    Counts the number of pixels in an image that are undersaturated (below a threshold).

    Args:
      img: An image obkject read in by open cv
      threshold: The threshold value below which a pixel is considered undersaturated (0-255).

    Returns:
      int: The number of undersaturated pixels in the image.
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Get the saturation channel (S)
    saturation = hsv[:, :, 1]

    # Count pixels below the threshold
    undersaturated_pixels = (saturation < threshold).sum()

    undersaturated_percentage = (undersaturated_pixels / saturation.size) * 100

    return round(undersaturated_percentage, 1)


def count_oversaturated_pixels(img: np.ndarray, threshold: int = 235) -> float:
    """
    Counts the number of pixels in an image that are oversaturated (above a threshold).

    Args:
      img: Path to the image file.
      threshold: The threshold value above which a pixel is considered oversaturated (0-255).

    Returns:
      int: The number of oversaturated pixels in the image.
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Get the saturation channel (S)
    saturation = hsv[:, :, 1]

    # Count pixels above the threshold
    oversaturated_pixels = (saturation > threshold).sum()

    # Calculate the percentage of oversaturated pixels
    oversaturation_percentage = (oversaturated_pixels / saturation.size) * 100

    return round(oversaturation_percentage, 1)
