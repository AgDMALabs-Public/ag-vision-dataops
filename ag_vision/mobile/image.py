import numpy as np
from ag_vision.core.image import AgImage
from open_aglabs.image.models import Image
import logging

class MobileImage(AgImage):
    def __init__(self, platform: str = None, cloud_bucket: str = None, img_key: str = None,
                 metadata_key: str = None, image: np.ndarray or None = None, metadata: Image or None = None):
        # Call the initializer of the parent class 'Image' to handle the common attributes
        super().__init__(platform, cloud_bucket, img_key, metadata_key, image, metadata)

