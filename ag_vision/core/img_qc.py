import cv2
import numpy as np
from PIL import Image
import tempfile

import torch
import torchvision
from torchvision import transforms

from huggingface_hub import hf_hub_download


def convert_opencv_to_pil(opencv_image):
    """
    Converts an OpenCV image (NumPy array) to a PIL Image.

    Args:
        opencv_image: The image loaded using OpenCV (in BGR format).

    Returns:
        A PIL Image object in RGB format.
    """
    # 1. Convert the color from BGR to RGB
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    # 2. Create a PIL Image from the NumPy array
    pil_image = Image.fromarray(rgb_image)

    return pil_image


class BlurInference:
    def __init__(self, device=None, cache_dir=None, local_dir = None):
        """
        Initializes the model, transforms, and other static objects for blur inference.
        This is the expensive setup that should only be run once.
        """
        self.device = device if device else torch.device("cpu")
        self.class_names = sorted([str(i) for i in range(0, 11)])
        num_classes = len(self.class_names)

        HF_MODEL_REPO_ID = "dwilli37/ag_image_blur_detection"
        HF_WEIGHTS_FILENAME = "blur_weights_2.pth"

        # 1. Initialize the model architecture
        inference_model = torchvision.models.resnet50(weights=None)
        num_ftrs = inference_model.fc.in_features
        inference_model.fc = torch.nn.Linear(num_ftrs, num_classes)
        inference_model.to(self.device)

        # 2. Download weights from Hugging Face Hub and load them
        try:
            # This function handles download, caching, and returns the local path
            model_path = hf_hub_download(repo_id=HF_MODEL_REPO_ID,
                                         filename=HF_WEIGHTS_FILENAME,
                                         cache_dir=cache_dir,
                                         local_dir=local_dir)

            print(f"Loaded weights from local cache: {model_path}")

            # Load the state dict using the path returned by hf_hub_download
            inference_model.load_state_dict(
                torch.load(model_path, map_location=self.device))

        except Exception as e:
            # Handle potential errors (e.g., file not found, connection error)
            print(f"ERROR: Could not load model weights from Hugging Face Hub.")
            print(f"Please check REPO_ID '{HF_MODEL_REPO_ID}' and filename '{HF_WEIGHTS_FILENAME}'.")
            raise RuntimeError(f"Error in BlurInference.__init__: {e}") from e

        self.model = inference_model.eval()  # Set to evaluation mode

        # 3. Define the image transformation pipeline ONCE
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, cv_img: np.ndarray) -> dict:
        """
        Runs fast inference on a single OpenCV image using the pre-loaded model.
        This method is designed to be called repeatedly in a loop.
        """
        # 1. Preprocess the image
        img_pil = convert_opencv_to_pil(cv_img)
        input_tensor = self.transform(img_pil)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        # 2. Run inference with torch.no_grad() to disable gradient calculations
        with torch.no_grad():
            output = self.model(input_batch)

        # 3. Post-process the output
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class_idx = torch.argmax(probabilities).item()

        return {
            'pred': self.class_names[predicted_class_idx],
            'prob': probabilities[predicted_class_idx].item(),
            'version': 2,
            'id': 'dwilli37/ag_image_blur_detection'
        }


class AgImageType:
    def __init__(self, device=None, cache_dir=None, local_dir = None):
        """
        Initializes the model, transforms, and other static objects for blur inference.
        This is the expensive setup that should only be run once.
        """
        self.device = device if device else torch.device("cpu")
        self.class_names = sorted(['half-plot', 'junk', 'multi-plant', 'single-plant', 'whole-plot', 'field'])
        num_classes = len(self.class_names)

        HF_MODEL_REPO_ID = "dwilli37/ag_image_type"
        HF_WEIGHTS_FILENAME = "version_1.pth"

        # 1. Initialize the model architecture
        inference_model = torchvision.models.resnet50(weights=None)
        num_ftrs = inference_model.fc.in_features
        inference_model.fc = torch.nn.Linear(num_ftrs, num_classes)
        inference_model.to(self.device)

        # 2. Download weights from Hugging Face Hub and load them
        try:
            # This function handles download, caching, and returns the local path
            model_path = hf_hub_download(repo_id=HF_MODEL_REPO_ID,
                                         filename=HF_WEIGHTS_FILENAME,
                                         cache_dir=cache_dir,
                                         local_dir=local_dir)

            print(f"Loaded weights from local cache: {model_path}")

            # Load the state dict using the path returned by hf_hub_download
            inference_model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )

        except Exception as e:
            # Handle potential errors (e.g., file not found, connection error)
            print(f"ERROR: Could not load model weights from Hugging Face Hub.")
            print(f"Please check REPO_ID '{HF_MODEL_REPO_ID}' and filename '{HF_WEIGHTS_FILENAME}'.")
            raise RuntimeError(f"Error in AgImageType.__init__: {e}") from e


        self.model = inference_model.eval()  # Set to evaluation mode

        # 3. Define the image transformation pipeline ONCE
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, cv_img: np.ndarray) -> dict:
        """
        Runs fast inference on a single OpenCV image using the pre-loaded model.
        This method is designed to be called repeatedly in a loop.
        """
        # 1. Preprocess the image
        img_pil = convert_opencv_to_pil(cv_img)
        input_tensor = self.transform(img_pil)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        # 2. Run inference with torch.no_grad() to disable gradient calculations
        with torch.no_grad():
            output = self.model(input_batch)

        # 3. Post-process the output
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class_idx = torch.argmax(probabilities).item()

        return {
            'pred': self.class_names[predicted_class_idx],
            'prob': probabilities[predicted_class_idx].item(),
            'version': 1,
            'id': 'dwilli37/ag_image_type'
        }


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
