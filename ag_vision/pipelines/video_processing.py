import pandas as pd
import logging
import os
from ag_vision.core.video import AgVideo
from ag_vision.constants import paths as pth

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_single_video(image_id, file_path, platform: str):
    metadata_file = pth.generate_metadata_path_from_file_name(file_path)

    try:
        ag_img = AgVideo(metadata_key=metadata_file,
                         platform=platform)
        ag_img.read_metadata()

        if ag_img.metadata.image_quality is None:
            return {'image_id': image_id, 'error': 'No Image Quality Metadata'}

        vq = ag_img.metadata.video_quality
        loc = ag_img.metadata.location_properties

        return {
            'image_id': image_id,
            'metadata_path': metadata_file,
            'frame_rate': float(vq.frame_rate),
            'frames': float(vq.frames),
            'height_pxl': int(vq.height),
            'width_pxl': int(vq.width),
            'orientation': str(vq.orientation),
            'rotation': float(vq.rotation),
            'latitude': float(loc.latitude),
            'longitude': float(loc.longitude),
            'error': ""
        }

    except Exception as e:
        return {
            'image_id': image_id,
            'metadata_path': metadata_file,
            'frame_rate': 0.0,
            'frames': 0.0,
            'height_pxl': 0,
            'width_pxl': 0,
            'orientation': "",
            'rotation': -1,
            'latitude': 0,
            'longitude': -1,
            'error': str(e)
        }


def generate_video_table(img_list: list, platform: str, project_index: int = 5) -> pd.DataFrame:
    """

    """
    img_list = list(img_list)

    img_df = pd.DataFrame({'file_path': img_list})

    img_df['image_id'] = [os.path.basename(x).split('.')[0] for x in img_list]
    img_df['project'] = [x.split('/')[project_index] for x in img_list]
    img_df['site'] = [x.split('/')[project_index + 1] for x in img_list]
    img_df['trial'] = [x.split('/')[project_index + 2] for x in img_list]
    img_df['season'] = [x.split('/')[project_index + 3] for x in img_list]
    img_df['field'] = [x.split('/')[project_index + 4] for x in img_list]
    img_df['location'] = [x.split('/')[project_index + 5] for x in img_list]
    img_df['protocol'] = [x.split('/')[project_index + 8] for x in img_list]
    img_df['upload_date'] = [x.split('/')[project_index + 9] for x in img_list]

    img_df['plot_id'] = [
        x.split('/')[project_index + 9]
        if x.split('/')[project_index + 9] != os.path.basename(x).split('.')[0]
        else 'none'
        for x in img_list
    ]

    rows = list(img_df[['image_id', 'file_path']].itertuples(index=False))
    out_list = []
    for row in rows:
        out_list.append(process_single_video(image_id=row[0],
                                             file_path=row[1],
                                             platform=platform))

    final_df = pd.DataFrame(out_list)

    video_df = pd.merge(img_df, final_df, on='image_id', how='left')

    return video_df


def generate_metadata_files_from_video_list(file_paths: list, platform: str, cloud_bucket: str = None,
                                            image_type: str = None) -> pd.DataFrame:
    """

    """
    file_paths = list(file_paths)
    out_df_list = []

    for file_path in file_paths:
        df = pd.DataFrame({'file_path': [file_path],
                           'status': 'unknown'})
        try:
            ag_vid = AgVideo(video_key=file_path,
                             platform=platform,
                             cloud_bucket=cloud_bucket)

            ag_vid.generate_metadata_key_from_img_key()

            ag_vid.initialize_metadata(img_type=image_type)

            ag_vid.extract_metadata_from_exif()

            ag_vid.save_metadata()
            df['status'] = 'success'
            out_df_list.append(df)

        except Exception as e:
            print(f'Fail, {e}')
            df['status'] = str(e)
            out_df_list.append(df)

    out_df = pd.concat(out_df_list)

    return out_df


def update_exif_metadata_from_video_list(file_paths: list, platform: str, cloud_bucket: str = None) -> pd.DataFrame:
    """

    """
    file_paths = list(file_paths)
    out_df_list = []

    for file_path in file_paths:
        df = pd.DataFrame({'file_path': [file_path],
                           'status': 'unknown'})
        try:
            ag_vid = AgVideo(video_key=file_path,
                             platform=platform,
                             cloud_bucket=cloud_bucket)

            ag_vid.generate_metadata_key_from_img_key()

            ag_vid.read_metadata()

            ag_vid.extract_metadata_from_exif()

            ag_vid.save_metadata()
            df['status'] = 'success'
            out_df_list.append(df)

        except Exception as e:
            print(f'Fail, {e}')
            df['status'] = str(e)
            out_df_list.append(df)

    out_df = pd.concat(out_df_list)

    return out_df
