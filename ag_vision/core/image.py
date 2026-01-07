import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from open_aglabs.image.models import AgImageModel
from open_aglabs.core import base_models as bm
from open_aglabs.core import constants as cst
from open_aglabs.image import models as im

from ag_vision.data_io import aws_io, local_io, databricks_io
from ag_vision.core import img_qc as iq
from ag_vision.constants import paths
from ag_vision.data_io import schemas as sch
import logging

logger = logging.getLogger(__name__)  # Use __name__ to get the module's name


class AgImage:
    def __init__(self, platform: str = None, cloud_bucket: str = None, img_key: str = None,
                 metadata_key: str = None, image: np.ndarray or None = None, metadata: AgImageModel or None = None):
        """

        """
        if not (image is None or isinstance(image, np.ndarray)):
            raise TypeError("image must be an instance of np.ndarray (opencv format)")

        self.platform = platform
        self.cloud_bucket = cloud_bucket
        self.img_key = img_key
        self.metadata_key = metadata_key
        self.image = image
        self.metadata = metadata
        self.exif = None

    def __repr__(self):
        return f"FieldImage(key={self.img_key},image={self.image}, metadata={self.metadata})"

    def generate_metadata_key_from_img_key(self):
        self.metadata_key = paths.generate_metadata_path_from_file_name(data_path=self.img_key)

    def read_img(self):
        if self.platform == 'db':
            try:
                self.image = databricks_io.read_img_from_databricks(file_name=self.img_key)
            except Exception as e:
                self.image = np.ndarray([])
                logger.warning(f'Exception occurred while reading image from DBFS: {e}')

        elif self.platform == 'local':
            try:
                self.image = local_io.read_image(image_path=self.img_key)
            except Exception as e:
                self.image = np.ndarray([])
                logger.warning(f'Exception occurred while reading image from local: {e}')
        else:
            logger.warning(f'The cloud platform need to be local or db')

    def save_img(self):
        if self.image is not None and self.img_key is not None:
            if self.platform == 'db':
                try:
                    databricks_io.save_img_to_databricks(img=self.image,
                                                         file_name=self.img_key)
                except Exception as e:
                    logger.warning(f'Exception occurred while saving image to DBFS: {e}')

            elif self.platform == 'local':
                try:
                    local_io.save_image(image=self.image,
                                        save_path=self.img_key)
                except Exception as e:
                    logger.warning(f'Exception occurred while saving image to local: {e}')
            else:
                logger.warning(f'The cloud platform need to be local or db')

        else:
            logger.warning(
                f'One of the following attributes is None: image: {self.image}, cloud_key: {self.cloud_img_key}')

    def read_metadata(self):
        if self.platform == 'db':
            if os.path.exists(self.metadata_key):
                try:
                    m_data = databricks_io.read_json_from_databricks(file_name=self.metadata_key)
                    self.metadata = AgImageModel(**m_data)
                except Exception as e:
                    self.metadata = None
                    logger.warning(f'Exception occurred while reading metadata from DBFS: {e}')

        elif self.platform == 'local':
            try:
                m_data = local_io.read_json(file_path=self.metadata_key)
                self.metadata = AgImageModel(**m_data)
            except Exception as e:
                self.metadata = None
                logger.warning(f'Exception occurred while reading metadata from local: {e}')
        else:
            logger.warning(f'The cloud platform need to be local or db')

    def load_metadata_from_dict(self, metadata_dict: dict):
        try:
            self.metadata = AgImageModel(**metadata_dict)
        except Exception as e:
            self.metadata = None
            logger.warning(f'Exception occurred while reading metadata from DBFS: {e}')

    def save_metadata(self):
        if self.metadata is not None and self.metadata_key is not None:
            if self.platform == 'db':
                try:
                    databricks_io.save_json_to_databricks(data=self.metadata.model_dump(),
                                                          file_name=self.metadata_key)

                except Exception as e:
                    logger.warning(f'Exception occurred while saving metadata to DB: {e}')

            elif self.platform == 'local':
                try:
                    local_io.save_json(data=self.metadata.model_dump(),
                                       file_path=self.metadata_key)
                except Exception as e:
                    logger.warning(f'Exception occurred while saving metadata to local: {e}')

        else:
            logger.warning(
                f'One of the following attributes is None: metadata: {self.metadata}, cloud_metadata_key: {self.metadata_key}')

    def upload_metadata_to_cloud(self, db_workspace_client):
        if self.metadata is not None and self.metadata_key is not None:
            if self.platform == 'db':
                try:
                    databricks_io.upload_json_to_databricks(w=db_workspace_client,
                                                            data=self.metadata.model_dump(),
                                                            file_name=self.metadata_key)

                except Exception as e:
                    logger.warning(f'Exception occurred while saving metadata to DB: {e}')
        else:
            logger.warning(
                f'One of the following attributes is None: metadata: {self.metadata}, cloud_metadata_key: {self.metadata_key}')

    def initialize_metadata(self, device: str = None, img_type: str = None):
        if self.img_key:
            metadata_dict = {
                "path": self.img_key,
                "id": self.img_key.split('/')[-1].split('.')[0],
                "device": device,
                "type": img_type,
                "protocol_properties": {},
                "camera_properties": {},
                "location_properties": {},
                "acquisition_properties": {},
                "image_quality": {}
            }
            self.metadata = AgImageModel(**metadata_dict)
        else:
            logger.warning(f'The image key is None and is needed to initialize the metadata.')

    def return_flattened_metadata(self):
        return sch.flatten_schema(schema_model=self.metadata,
                                  sep=':')

    def load_nested_metadata_from_flat(self, flat_metadata, sep: str = ':'):
        self.metadata = sch.generate_nested_schema_from_flat(schema_dict=flat_metadata,
                                                             sep=sep)

    # -------------------------------------Simple ways to view and image-----------------------------------------------#
    def plot_image(self, figsize=(10, 10)):
        """
        Plots the image stored in the object in RGB format using Matplotlib.

        Parameters:
        figsize (tuple, optional): Specifies the size of the figure in inches. Default is (10, 10).

        Behavior:
        - Converts the image from BGR to RGB format using OpenCV as Matplotlib expects RGB.
        - Displays the image using Matplotlib's imshow function.
        - Configures the layout to reduce padding using tight_layout.
        - Opens an interactive window for displaying the image using Matplotlib's show method.
        """
        try:
            plt.figure(figsize=figsize)
            # Since cv2 is in BGR format not RGB...
            plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.warning(f'Exception occurred while plotting image: {e}')

    # ----------------------------------Below are the methods to deal with metadata IO --------------------------------#
    def add_custom_data_to_metadata(self, key: str, value: str, overwrite: bool = False):
        self.metadata = sch.add_first_level_nested_metadata(metadata_model=self.metadata,
                                                            root_model=bm.Other(),
                                                            root_key='other',
                                                            first_level_key=key,
                                                            first_level_value=value,
                                                            overwrite=overwrite)

    # --------------------------Below are the methods to deal with base image metadata --------------------------------#
    def add_image_id_to_metadata(self, overwrite: bool = False):
        if self.img_key:
            self.metadata = sch.add_data_to_model(model_obj=self.metadata,
                                                  key='id',
                                                  value=self.img_key.split('/')[-1].split('.')[0],
                                                  overwrite=overwrite)
        else:
            print("No image path or cloud key found to generate the image id.")

    def add_device_to_metadata(self, device_name: str, overwrite: bool = False):
        assert device_name.lower() in cst.DEVICE_LIST, f"The device options are {cst.DEVICE_LIST}"
        self.metadata = sch.add_data_to_model(model_obj=self.metadata,
                                              key='device',
                                              value=device_name,
                                              overwrite=overwrite)

    def add_image_type_to_metadata(self, img_type: str, overwrite: bool = False):
        assert img_type.lower() in cst.IMAGE_TYPE_LIST, f"The type options are {cst.IMAGE_TYPE_LIST}"
        self.metadata = sch.add_data_to_model(model_obj=self.metadata,
                                              key='type',
                                              value=img_type,
                                              overwrite=overwrite)

    # -----------------------------METADATA IMAGE QUALITY METHODS------------------------------------------------------#
    def add_image_quality_exposure_to_metadata(self, overwrite: bool = False):
        """
        Adds image quality exposure information to the metadata.

        This method calculates the relative exposure of an image and adds it to the metadata under the
        'image_quality -> exposure' key structure. The addition is handled via the
        `_add_first_level_nested_metadata` method.

        Arguments:
        overwrite (bool): Determines whether existing metadata should be overwritten. Defaults to False.
        """
        self.metadata = sch.add_first_level_nested_metadata(metadata_model=self.metadata,
                                                            root_model=im.ImageQuality(),
                                                            root_key='image_quality',
                                                            first_level_key='exposure',
                                                            first_level_value=iq.calculate_relative_exposure(
                                                                img=self.image),
                                                            overwrite=overwrite)

    def add_image_quality_height_width_orientation_to_metadata(self, overwrite: bool = False):
        """
        Adds image quality metadata including height, width, and orientation to the metadata object.

        This method fetches the image's height and width from the image data and computes its orientation
        (landscape or portrait). It then adds these values to the metadata under the 'image_quality' root key.

        Parameters:
        overwrite (bool): Indicates whether existing metadata with the same keys should be overwritten.

        Raises:
        None
        """
        height, width = self.image.shape[:2]  # OpenCV returns shape as (height, width, channels)

        self.metadata = sch.add_first_level_nested_metadata(metadata_model=self.metadata,
                                                            root_model=im.ImageQuality(),
                                                            root_key='image_quality',
                                                            first_level_key='height',
                                                            first_level_value=height,
                                                            overwrite=overwrite)

        self.metadata = sch.add_first_level_nested_metadata(metadata_model=self.metadata,
                                                            root_model=im.ImageQuality(),
                                                            root_key='image_quality',
                                                            first_level_key='width',
                                                            first_level_value=width,
                                                            overwrite=overwrite)

        self.metadata = sch.add_first_level_nested_metadata(metadata_model=self.metadata,
                                                            root_model=im.ImageQuality(),
                                                            root_key='image_quality',
                                                            first_level_key='orientation',
                                                            first_level_value='landscape' if width >= height else 'portrait',
                                                            overwrite=overwrite)

    def add_image_quality_saturation_metrics_to_metadata(self, overwrite: bool = False):
        """
        Adds image quality saturation metrics to the metadata of an image.

        This function computes the percentage of pixels in the image that are oversaturated and undersaturated.
        The calculated metrics are stored in the metadata under the 'image_quality' root key.

        Parameters:
        overwrite (bool): Determines whether to overwrite existing metadata values if the keys already exist.
        """
        self.metadata = sch.add_first_level_nested_metadata(metadata_model=self.metadata,
                                                            root_model=im.ImageQuality(),
                                                            root_key='image_quality',
                                                            first_level_key='pct_pixel_over_saturation',
                                                            first_level_value=iq.count_oversaturated_pixels(
                                                                img=self.image,
                                                                threshold=254),
                                                            overwrite=overwrite)

        self.metadata = sch.add_first_level_nested_metadata(metadata_model=self.metadata,
                                                            root_model=im.ImageQuality(),
                                                            root_key='image_quality',
                                                            first_level_key='pct_pixel_under_saturation',
                                                            first_level_value=iq.count_undersaturated_pixels(
                                                                img=self.image,
                                                                threshold=5),
                                                            overwrite=overwrite)

    def add_image_quality_blur_metrics_to_metadata(self, pred: float, confidence: float, version: str, model_id: str,
                                                   overwrite: bool = False):
        """
        Adds blur metrics related to image quality to the metadata.

        Arguments:
        pred : float
            Predicted value for the blur metric.
        model_name : str
            Name of the model used to generate the blur metric.
        overwrite : bool, optional
            Flag to indicate if existing metadata should be overwritten. Default is False.
        """
        self.metadata = sch.add_first_level_nested_model_metadata(metadata_model=self.metadata,
                                                                  root_model=im.ImageQuality(),
                                                                  root_key='image_quality',
                                                                  first_level_model=bm.MLOutput(),
                                                                  first_level_key='blur_score',
                                                                  pred=pred,
                                                                  confidence=confidence,
                                                                  model_id=model_id,
                                                                  version=version,
                                                                  overwrite=overwrite)

    def add_image_quality_blur_metrics_to_metadata(self, pred: float, confidence: float, version: str, model_id: str,
                                                   overwrite: bool = False):
        """
        Adds blur metrics related to image quality to the metadata.

        Arguments:
        pred : float
            Predicted value for the blur metric.
        model_name : str
            Name of the model used to generate the blur metric.
        overwrite : bool, optional
            Flag to indicate if existing metadata should be overwritten. Default is False.
        """
        self.metadata = sch.add_first_level_nested_model_metadata(metadata_model=self.metadata,
                                                                  root_model=im.ImageQuality(),
                                                                  root_key='image_quality',
                                                                  first_level_model=bm.MLOutput(),
                                                                  first_level_key='blur_score',
                                                                  pred=pred,
                                                                  confidence=confidence,
                                                                  model_id=model_id,
                                                                  version=version,
                                                                  overwrite=overwrite)

    def add_acq_properties_object_resolution_ml_metrics_to_metadata(self,
                                                                    pred: float,
                                                                    confidence: float,
                                                                    version: str,
                                                                    model_id: str,
                                                                    overwrite: bool = False):
        """
        Adds information about the object resolutions in the images.

        Arguments:
        pred : float
            Predicted value for the blur metric.
        model_name : str
            Name of the model used to generate the blur metric.
        overwrite : bool, optional
            Flag to indicate if existing metadata should be overwritten. Default is False.
        """
        self.metadata = sch.add_first_level_nested_model_metadata(metadata_model=self.metadata,
                                                                  root_model=im.AcquisitionProperties(),
                                                                  root_key='acquisition_properties',
                                                                  first_level_model=bm.MLOutput(),
                                                                  first_level_key='object_resolution_ml',
                                                                  pred=pred,
                                                                  confidence=confidence,
                                                                  model_id=model_id,
                                                                  version=version,
                                                                  overwrite=overwrite)

    # -----------------------------Below are the methods to deal with IMAGE AQUISITION------------------------------------#
    def add_image_acquisition_properties_to_metadata(self, date: str, height_m: float, angle: float, light_source: str,
                                                     setting: str, overwrite: bool = False):
        """
        """
        if date:
            self.metadata = sch.add_first_level_nested_metadata(metadata_model=self.metadata,
                                                                root_model=im.AcquisitionProperties(),
                                                                root_key='acquisition_properties',
                                                                first_level_key='date',
                                                                first_level_value=date,
                                                                overwrite=overwrite)

        if height_m:
            self.metadata = sch.add_first_level_nested_metadata(metadata_model=self.metadata,
                                                                root_model=im.AcquisitionProperties(),
                                                                root_key='acquisition_properties',
                                                                first_level_key='camera_height_m',
                                                                first_level_value=height_m,
                                                                overwrite=overwrite)

        if angle:
            self.metadata = sch.add_first_level_nested_metadata(metadata_model=self.metadata,
                                                                root_model=im.AcquisitionProperties(),
                                                                root_key='acquisition_properties',
                                                                first_level_key='camera_angle_deg',
                                                                first_level_value=angle,
                                                                overwrite=overwrite)

        if light_source:
            self.metadata = sch.add_first_level_nested_metadata(metadata_model=self.metadata,
                                                                root_model=im.AcquisitionProperties(),
                                                                root_key='acquisition_properties',
                                                                first_level_key='light_source',
                                                                first_level_value=light_source,
                                                                overwrite=overwrite)

        if setting:
            self.metadata = sch.add_first_level_nested_metadata(metadata_model=self.metadata,
                                                                root_model=im.AcquisitionProperties(),
                                                                root_key='acquisition_properties',
                                                                first_level_key='setting',
                                                                first_level_value=setting,
                                                                overwrite=overwrite)
