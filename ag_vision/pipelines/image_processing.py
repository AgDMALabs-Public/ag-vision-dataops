import pandas as pd
import logging
from tqdm import tqdm
import os
from ag_vision.core.image import AgImage
from ag_vision.constants import paths as pth
from ag_vision.core import img_qc as iq

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_single_image(image_id, file_path, platform: str):
    metadata_file = pth.generate_metadata_path_from_file_name(file_path)

    try:
        ag_img = AgImage(metadata_key=metadata_file,
                         platform=platform)
        ag_img.read_metadata()

        if ag_img.metadata.image_quality is None:
            return {'image_id': image_id, 'error': 'No Image Quality Metadata'}

        iq = ag_img.metadata.image_quality
        acq = ag_img.metadata.acquisition_properties

        return {
            'image_id': image_id,
            'metadata_path': metadata_file,
            'pct_pixel_over_saturation': float(iq.pct_pixel_over_saturation),
            'pct_pixel_under_saturation': float(iq.pct_pixel_under_saturation),
            'height_pxl': int(iq.height),
            'width_pxl': int(iq.width),
            'orientation': str(iq.orientation),
            'exposure': float(iq.exposure),
            'blur': int(iq.blur_score.pred) if iq.blur_score else -1,
            'ml_object_resolution': str(acq.object_resolution_ml.pred) if acq.object_resolution_ml else "",
            'error': ""
        }

    except Exception as e:
        return {
            'image_id': image_id,
            'metadata_path': metadata_file,
            'pct_pixel_over_saturation': 0.0,
            'pct_pixel_under_saturation': 0.0,
            'height_pxl': 0,
            'width_pxl': 0,
            'orientation': "",
            'exposure': 0,
            'blur': -1,
            'ml_object_resolution': "",
            'error': str(e)
        }


def generate_images_table(img_list: list, platform: str, project_index: int = 6) -> pd.DataFrame:
    """

    """
    img_list = list(img_list)

    img_df = pd.DataFrame({'file_path': img_list})

    img_df['image_id'] = [os.path.basename(x).split('.')[0] for x in img_list]
    img_df['project'] = [x.split('/')[project_index] for x in img_list]
    img_df['trial'] = [x.split('/')[project_index + 1] for x in img_list]
    img_df['season'] = [x.split('/')[project_index + 2] for x in img_list]
    img_df['field'] = [x.split('/')[project_index + 3] for x in img_list]
    img_df['location'] = [x.split('/')[project_index + 4] for x in img_list]
    img_df['protocol'] = [x.split('/')[project_index + 7] for x in img_list]
    img_df['upload_date'] = [x.split('/')[project_index + 8] for x in img_list]

    img_df['plot_id'] = [
        x.split('/')[project_index + 9]
        if x.split('/')[project_index + 9] != os.path.basename(x).split('.')[0]
        else 'none'
        for x in img_list
    ]

    rows = list(img_df[['image_id', 'file_path']].itertuples(index=False))
    out_list = []
    for row in tqdm(rows):
        out_list.append(process_single_image(image_id=row[0],
                                             file_path=row[1],
                                             platform=platform))

    final_df = pd.DataFrame(out_list)

    img_df = pd.merge(img_df, final_df, on='image_id', how='left')

    return img_df


def add_core_metadata_to_ag_image(ag_image: AgImage) -> None:
    """

    """
    ag_image.add_image_quality_saturation_metrics_to_metadata()
    ag_image.add_image_quality_height_width_orientation_to_metadata()
    ag_image.add_image_quality_exposure_to_metadata()


def generate_metadata_files_from_image_list(file_paths: list, platform: str, cloud_bucket: str = None,
                                          image_type: str = None) -> None:
    """

    """
    file_paths = list(file_paths)

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

    for file_path in tqdm(file_paths):
        try:
            ag_img = AgImage(img_key=file_path,
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


def update_ml_metadata(file_paths, platform: str, cloud_bucket: str = None, blur_class = None, ait_class= None, model_cache_dir=None, model_local_dir: None = None):
    be = 'success'
    me = 'success'
    if blur_class is None:
        try:
            blur_class = iq.BlurInference(cache_dir=model_cache_dir,
                                          local_dir=model_local_dir)
        except Exception as e:
            be = str(e)
            blur_class = None

    if ait_class is None:
        try:
            ait_class = iq.AgImageType(cache_dir=model_cache_dir,
                                       local_dir=model_local_dir)
        except Exception as e:
            me = str(e)
            ait_class = None

    out_df_list = []

    for file_path in tqdm(file_paths):
        df = pd.DataFrame({'file_path': [file_path],
                           'status': 'unknown'})
        try:
            ag_img = AgImage(img_key=file_path,
                             platform=platform,
                             cloud_bucket=cloud_bucket)

            ag_img.read_img()
            ag_img.generate_metadata_key_from_img_key()
            ag_img.read_metadata()

            if blur_class is not None:
                df['status'] = 'success'
                blur_inf = blur_class.predict(cv_img=ag_img.image)
                ag_img.add_image_quality_blur_metrics_to_metadata(pred=blur_inf['pred'],
                                                                  confidence=blur_inf['prob'],
                                                                  version=blur_inf['version'],
                                                                  model_id=blur_inf['id'],
                                                                  overwrite=True)
            if ait_class is not None:
                df['status'] = 'success'
                ait_inf = ait_class.predict(cv_img=ag_img.image)
                ag_img.add_acq_properties_object_resolution_ml_metrics_to_metadata(pred=ait_inf['pred'],
                                                                                   confidence=ait_inf['prob'],
                                                                                   version=ait_inf['version'],
                                                                                   model_id=ait_inf['id'],
                                                                                   overwrite=True)

            ag_img.save_metadata()

            out_df_list.append(df)

        except Exception as e:
            print(f'Fail, {e}')
            df['status'] = str(e)
            out_df_list.append(df)

    out_df = pd.concat(out_df_list)
    out_df['be'] = be
    out_df['me'] = me
    return out_df
