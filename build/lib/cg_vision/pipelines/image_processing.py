import pandas as pd
import logging
from tqdm import tqdm
import os
from ag_vision.core.image import AgImage
from ag_vision.constants import paths as pth
from ag_vision.core import img_qc as iq

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_images_table(img_list: list, platform: str, project_index: int = 6):
    """

    """
    img_df = pd.DataFrame({'file_path': img_list})

    img_df['image_id'] = [os.path.basename(x).split('.')[0] for x in img_list]
    img_df['project'] = [x.split('/')[project_index] for x in img_list]
    img_df['trial'] = [x.split('/')[project_index + 1] for x in img_list]
    img_df['season'] = [x.split('/')[project_index + 2] for x in img_list]
    img_df['field'] = [x.split('/')[project_index + 3] for x in img_list]
    img_df['location'] = [x.split('/')[project_index + 4] for x in img_list]
    img_df['protocol'] = [x.split('/')[project_index + 7] for x in img_list]
    img_df['upload_date'] = [x.split('/')[project_index + 8] for x in img_list]

    img_df['plot_id'] = new_list = [
        x.split('/')[project_index + 9]
        if x.split('/')[project_index + 9] != os.path.basename(x).split('.')[0]
        else 'none'
        for x in img_list
    ]

    for row in tqdm(img_df.itertuples(), total=len(img_df)):
        idx = row[0]
        metadata_file = pth.generate_metadata_path_from_file_name(row[1])

        if os.path.exists(metadata_file):
            try:
                ag_img = AgImage(metadata_key=metadata_file,
                                 platform=platform)

                ag_img.read_metadata()

                if ag_img.metadata.image_quality is None:
                    continue

                img_df.loc[idx, 'metadata_path'] = metadata_file

                if ag_img.metadata.image_quality is not None:
                    img_df.loc[
                        idx, 'pct_pixel_over_saturation'] = ag_img.metadata.image_quality.pct_pixel_over_saturation
                    img_df.loc[
                        idx, 'pct_pixel_under_saturation'] = ag_img.metadata.image_quality.pct_pixel_under_saturation
                    img_df.loc[idx, 'height_pxl'] = ag_img.metadata.image_quality.height
                    img_df.loc[idx, 'width_pxl'] = ag_img.metadata.image_quality.width
                    img_df.loc[idx, 'orientation'] = ag_img.metadata.image_quality.orientation
                    img_df.loc[idx, 'exposure'] = ag_img.metadata.image_quality.exposure

                    if ag_img.metadata.image_quality.blur_score is not None:
                        img_df.loc[idx, 'blur'] = ag_img.metadata.image_quality.blur_score.pred

                    if ag_img.metadata.acquisition_properties.object_resolution_ml is not None:
                        img_df.loc[idx, 'ml_object_resolution'] = ag_img.metadata.acquisition_properties.object_resolution_ml.pred

            except Exception as e:
                img_df.loc[idx, 'error'] = str(e)

    return img_df


def add_core_metadata_to_ag_image(ag_image: AgImage) -> None:
    """

    """
    ag_image.add_image_quality_saturation_metrics_to_metadata()
    ag_image.add_image_quality_height_width_orientation_to_metadata()
    ag_image.add_image_quality_exposure_to_metadata()


def generate_metadata_files_from_image_df(in_df, platform: str, cloud_bucket: str = None, image_type: str = None) -> None:
    """

    """
    try:
        blur = iq.BlurInference()
    except Exception as e:
        print(f'Failed to load blur model: {e}')
        blur = None

    try:
        ait_model = iq.AgImageType()
    except Exception as e:
        print(f'Failed to load AIT model: {e}')
        ait_model = None

    for row in tqdm(in_df.itertuples(), total=len(in_df)):
        try:
            ag_img = AgImage(img_key=row.file_path,
                             platform=platform,
                             cloud_bucket=cloud_bucket)

            ag_img.generate_metadata_key_from_img_key()

            ag_img.initialize_metadata(img_type=image_type)

            ag_img.read_img()

            if blur is not None:
                blur_inf = blur.predict(cv_img=ag_img.image)
                ag_img.add_image_quality_blur_metrics_to_metadata(pred=blur_inf['pred'],
                                                                  confidence=blur_inf['prob'],
                                                                  version=blur_inf['version'],
                                                                  model_id=blur_inf['id'])
            if ait_model is not None:
                ait_inf = ait_model.predict(cv_img=ag_img.image)
                ag_img.add_acq_properties_object_resolution_ml_metrics_to_metadata(pred=ait_inf['pred'],
                                                                                   confidence=ait_inf['prob'],
                                                                                   version=ait_inf['version'],
                                                                                   model_id=ait_inf['id'])
            # add the core set of metadata we want.
            add_core_metadata_to_ag_image(ag_image=ag_img)

            ag_img.save_metadata()

        except Exception as e:
            print(f'Fail, {e}')
