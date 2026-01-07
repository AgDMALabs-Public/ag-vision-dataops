import logging
import cv2
import json
import os


logger = logging.getLogger(__name__)

def read_image(image_path: str):
    """
    Reads an image using OpenCV and returns it as a NumPy array.

    :param image_path: The file path to the image.
    :return: The image as a NumPy array (or None if the image can't be read).
    """
    # Read the image from the given path
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        logger.warning(f"Error: Unable to load image at {image_path}")
        return None

    return image


def save_image(image, save_path: str):
    """
    Saves the given image to the specified file path using OpenCV.

    :param image: The image to save (should be a NumPy array compatible with OpenCV).
    :param save_path: The path where the image will be saved (including the file name and extension).
    :return: True if the image is saved successfully, otherwise False.
    """
    try:
        success = cv2.imwrite(save_path, image)

        if success:
            logger.warning(f"Image successfully saved to {save_path}")
            return True
        else:
            logger.warning(f"Error: Failed to save image to {save_path}")
            return False
    except Exception as e:
        logger.warning(f"Exception occurred while saving the image: {e}")
        return False

def read_json(file_path: str) -> dict:
    """
    Reads data from a local JSON file.

    Args:
        file_path: The path to the local JSON file.

    Returns:
        A dictionary containing the JSON data,
        or an empty dictionary if an error occurs.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        logger.error(f"The file at {file_path} was not found.")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Could not decode JSON from the file at {file_path}.")
        return {}


def save_json(data: dict, file_path: str) -> bool:
    """
    Saves a dictionary to a local JSON file.

    Args:
        data: The dictionary to save.
        file_path: The path where the JSON file will be saved.

    Returns:
        True if the file is saved successfully, otherwise False.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON data successfully saved to {file_path}")
        return True
    except TypeError as e:
        logger.error(f"A TypeError occurred while serializing the dictionary to JSON: {e}")
        return False
    except IOError as e:
        logger.error(f"An IOError occurred while writing to {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return False
