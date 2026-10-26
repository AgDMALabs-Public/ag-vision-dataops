import logging
import cv2
import json
import os
import librosa
import soundfile as sf

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


def read_video(video_path: str) -> cv2.VideoCapture | None:
    """
    Reads a video file using OpenCV and returns a VideoCapture object.

    :param video_path: The file path to the video.
    :return: An opened cv2.VideoCapture object, or None if the video can't be read.
    """
    if not os.path.exists(video_path):
        logger.warning(f"Error: Unable to find video at {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.warning(f"Error: Unable to open video at {video_path}")
        return None

    return cap


def save_video(video: cv2.VideoCapture, save_path: str, fourcc: str = 'mp4v', fps: float = None) -> bool:
    """
    Saves a cv2.VideoCapture object to the specified file path frame-by-frame.

    :param video: An opened cv2.VideoCapture object to save.
    :param save_path: The path where the video will be saved (including file name and extension).
    :param fourcc: The 4-character codec code. Defaults to 'mp4v'.
    :param fps: Frames per second for the output. If None, uses the source video's FPS.
    :return: True if the video is saved successfully, otherwise False.
    """
    try:
        source_fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_fps = fps or source_fps

        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), out_fps, (width, height))

        while True:
            ret, frame = video.read()
            if not ret:
                break
            writer.write(frame)

        writer.release()
        logger.info(f"Video successfully saved to {save_path}")
        return True
    except Exception as e:
        logger.warning(f"Exception occurred while saving the video: {e}")
        return False


def read_json(file_path: str) -> dict:
    """
    Reads data from a local JSON file.
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
    """
    try:
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


def read_audio(file_path):
    """
    Reads audio file.

    Args:
        file_path: Path to the audio file.

    Returns:
        y: Audio signal.
        sr: Sampling rate.
    """
    try:
        y, sr = librosa.load(file_path)
    except Exception as e:
        y = None
        sr = None
        print(f'Failed to read audio file {e}')

    return y, sr


def save_audio(file_path, audio_data, sample_rate):
    """
    Saves audio data to a file.

    Args:
        file_path: The path to the file where the audio data will be saved.
        audio_data: The audio data to be saved.
        sample_rate: The sampling rate of the audio data.

    """
    try:
        sf.write(file_path, audio_data, sample_rate)
    except Exception as e:
        print(f'Failed to read audio file {e}')


def read_text_data(file_path: str, encoding:str='utf-8') -> str:
    """
    Reads the content of a text file.

    Args:
        file_path: The path to the .txt file.

    Returns:
        The content of the file as a string, or an empty string if an error occurs.
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"The file at {file_path} was not found.")
        return ""
    except Exception as e:
        logger.error(f"An error occurred while reading the file at {file_path}: {e}")
        return ""


def save_text_data(file_path: str, data: any) -> bool:
    """
    Saves data to a .txt file.

    Args:
        file_path: The path where the text file will be saved.
        data: The content to save (will be converted to a string).

    Returns:
        True if the file is saved successfully, otherwise False.
    """
    try:
        # Ensure the destination directory exists
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(str(data))

        logger.info(f"Text data successfully saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"An error occurred while saving the file to {file_path}: {e}")
        return False
