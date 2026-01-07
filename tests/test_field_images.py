import numpy as np
from ag_vision.core.image import AgImage
from open_aglabs.image.models import AgImageModel

data_dict = {
    "path": f"test/1231.jpg",
    "id": '1231',
    "device": "mobile",
    "type": "original",
    "camera_properties": {},
    "location_properties": {},
    "acquisition_properties": {},
    "image_quality": {}
}


def test_image_quality_metadata():
    # Setup
    image_data = np.random.randint(0, 256, (200, 150, 3), dtype=np.uint8)

    field_image = AgImage(image=image_data,
                             metadata=AgImageModel(**data_dict))

    # Call
    field_image.add_image_quality_exposure_to_metadata()
    field_image.add_image_quality_height_width_orientation_to_metadata()
    field_image.add_image_quality_blur_metrics_to_metadata(pred=10,
                                                           confidence=.9,
                                                           model_id="model_123",
                                                           version="v1")

    # Assert
    assert field_image.metadata.image_quality.exposure < 100
    assert field_image.metadata.image_quality.exposure > 1

    assert field_image.metadata.image_quality.height == 200
    assert field_image.metadata.image_quality.width == 150
    assert field_image.metadata.image_quality.orientation == 'portrait'

    assert field_image.metadata.image_quality.blur_score.model_id == "model_123"
    assert field_image.metadata.image_quality.blur_score.model_version == "v1"
    assert field_image.metadata.image_quality.blur_score.pred == 10
    assert field_image.metadata.image_quality.blur_score.confidence == 0.9


def test_base_image_values():
    image_data = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)

    field_image = AgImage(image=image_data,
                             metadata=AgImageModel(**data_dict))

    field_image.add_image_type_to_metadata(img_type="synthetic",
                                           overwrite=True)

    field_image.add_device_to_metadata(device_name='drone',
                                       overwrite=False)

    assert field_image.metadata.type == "synthetic"
    assert field_image.metadata.device == "mobile"


def test_image_acquisition_metadata():
    image_data = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)

    field_image = AgImage(image=image_data,
                             metadata=AgImageModel(**data_dict))

    field_image.add_image_acquisition_properties_to_metadata(date='1/1/2025',
                                                             height_m=10,
                                                             angle=45,
                                                             light_source='natural',
                                                             setting='open_field',
                                                             overwrite = False)

    field_image.add_image_acquisition_properties_to_metadata(date='1/1/2025',
                                                             height_m=11,
                                                             angle=50,
                                                             light_source='naturallly',
                                                             setting='open_field_1',
                                                             overwrite = False)
    print(field_image.metadata.acquisition_properties)
    assert field_image.metadata.acquisition_properties.date == '1/1/2025'
    assert field_image.metadata.acquisition_properties.camera_height_m == 10
    assert field_image.metadata.acquisition_properties.camera_angle_deg == 45
    assert field_image.metadata.acquisition_properties.light_source == 'natural'
    assert field_image.metadata.acquisition_properties.setting == 'open_field'


def test_add_image_id_to_metadata_from_cloud_img_key():
    # Setup
    field_image = AgImage(
        img_key="path/to/image_12345.jpg",
        metadata=AgImageModel(**data_dict)
    )

    # Call
    field_image.add_image_id_to_metadata()

    # Assert
    assert field_image.metadata.id == "1231"


def test_add_image_id_to_metadata_from_local_img_path():
    # Setup
    field_image = AgImage(
        img_key="images/dir/local_image_45678.png",
        metadata=AgImageModel(**data_dict)
    )

    # Call
    field_image.add_image_id_to_metadata(overwrite=True)

    # Assert
    assert field_image.metadata.id == "local_image_45678"

