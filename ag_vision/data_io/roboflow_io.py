import os
import shutil
import json
from tqdm import tqdm
import pandas as pd
from ag_vision.constants import paths
from ag_vision.data_io import annotation_io as aio
from ag_vision.data_io import databricks_io as dbio
from ag_vision.data_io import local_io as lio
from ag_vision.annotation import annotation as anno
import logging

logger = logging.getLogger(__name__)


def upload_image_to_roboflow(rf_project, batch_name: str, img_path: str, annotation_path: str = None,
                             split: str = 'train', tmp_copy: bool = True):
    if tmp_copy:
        temp_img_path = f"/tmp/{os.path.basename(img_path)}"
        os.makedirs(os.path.dirname(temp_img_path), exist_ok=True)
        shutil.copy(img_path, temp_img_path)

        if annotation_path is not None:
            temp_annotation_path = f"/tmp/{os.path.basename(annotation_path)}"
            os.makedirs(os.path.dirname(temp_annotation_path), exist_ok=True)
            shutil.copy(annotation_path, temp_annotation_path)
        else:
            tmp_annotation_path = None
    else:
        temp_img_path = img_path
        tmp_annotation_path = annotation_path

    rf_project.upload(
        image_path=temp_img_path,
        annotation_path=tmp_annotation_path,
        split=split,  # Optional: "train", "valid", or "test"
        batch_name=batch_name
    )


def upload_image_batch_to_roboflow_old(rf_project, annotation_type: str, project_path: str, task_name: str,
                                       batch_name: str, tmp_copy: bool = True,
                                       img_extension: list = ['.jpg', '.jpeg', '.tiff', '.png']):
    imgs_path = paths.annotation_image_path(project=project_path,
                                            annotation_type=annotation_type,
                                            task_name=task_name,
                                            batch_name=batch_name,
                                            f_name='none.jpg')

    img_dir = os.path.dirname(imgs_path)
    files = os.listdir(img_dir)
    imgs = []
    for f_name in files:
        if os.path.splitext(f_name)[1] in img_extension:
            imgs.append(img_dir + '/' + f_name)

    for img in tqdm(imgs):
        logging.info(f"Uploading {img} to Roboflow")
        upload_image_to_roboflow(rf_project=rf_project,
                                 batch_name=batch_name,
                                 img_path=img,
                                 annotation_path=None,
                                 split='train',
                                 tmp_copy=tmp_copy)


def upload_image_batch_to_roboflow(rf_project, annotation_type: str, project_path: str, task_name: str,
                                   batch_name: str, split: str, tmp_copy: bool = True,
                                   img_extension: list = ['.jpg', '.jpeg', '.tiff', '.png']):
    upload_annotation_batch_to_roboflow(rf_project=rf_project,
                                        annotation_type=annotation_type,
                                        project_path=project_path,
                                        task_name=task_name,
                                        batch_name=batch_name,
                                        download_date='',
                                        split=split,
                                        tmp_copy=tmp_copy,
                                        img_extension=img_extension,
                                        annotation=False)


def upload_annotation_batch_to_roboflow(rf_project, annotation_type: str, project_path: str, task_name: str,
                                        batch_name: str, download_date: str, split: str, tmp_copy: bool = True,
                                        img_extension: list = ['.jpg', '.jpeg', '.tiff', '.png'],
                                        annotation: bool = True):
    imgs_path = paths.annotation_image_path(project=project_path,
                                            annotation_type=annotation_type,
                                            task_name=task_name,
                                            batch_name=batch_name,
                                            f_name='none.jpg')

    img_dir = os.path.dirname(imgs_path)

    if annotation:
        annotation_path = paths.annotation_path(project=project_path,
                                                annotation_type=annotation_type,
                                                task_name=task_name,
                                                batch_name=batch_name,
                                                download_date=download_date,
                                                f_name='none.jpg')
        annotation_dir = os.path.dirname(annotation_path)

    files = os.listdir(img_dir)
    imgs = []

    for f_name in files:
        if os.path.splitext(f_name)[1] in img_extension:
            imgs.append(f_name)

    for img in tqdm(imgs):
        print(f"Uploading {img} to Roboflow")

        if annotation:
            annotation_file = annotation_dir + '/' + os.path.splitext(img)[0] + '.json'

            if os.path.exists(annotation_file):
                upload_image_to_roboflow(rf_project=rf_project,
                                         batch_name=batch_name,
                                         img_path=img_dir + '/' + img,
                                         annotation_path=annotation_file,
                                         split=split,
                                         tmp_copy=tmp_copy)
        else:
            upload_image_to_roboflow(rf_project=rf_project,
                                     batch_name=batch_name,
                                     img_path=img_dir + '/' + img,
                                     annotation_path=None,
                                     split=split,
                                     tmp_copy=tmp_copy)


def upload_annotation_df_to_roboflow(rf_project, annotation_df: pd.DataFrame, batch_name: str, img_col: str,
                                     annotation_col: str, split_col: str, tmp_copy: bool = True):
    """
    Uploads annotations and corresponding images from a DataFrame to a Roboflow project.

    This function iterates over each row in the given DataFrame and uploads the images and corresponding
    annotations to a specified Roboflow project. Additional parameters such as a batch name and optional
    temporary copy flag can be set to modify the behavior of the upload process.

    Args:
        rf_project: The target Roboflow project where the images and annotations will be uploaded.
        annotation_df (pd.DataFrame): DataFrame containing the image paths, annotation paths, and split
            information.
        batch_name (str): Name to be assigned to the batch upload in Roboflow.
        img_col (str): Name of the column in annotation_df that contains the file paths for the images.
        annotation_col (str): Name of the column in annotation_df that contains the file paths for the
            corresponding annotation files.
        split_col (str): Name of the column in annotation_df that contains the dataset split
            (e.g., train, valid, test).
        tmp_copy (bool, default=True): If True, creates a temporary copy of the files before uploading.

    Raises:
        Specific errors raised by upload_images_to_roboflow function during the upload
        process.
    """
    for idx, row in annotation_df.iterrows():
        logging.info(f"Uploading {row[img_col]} to Roboflow")

        upload_image_to_roboflow(rf_project=rf_project,
                                 batch_name=batch_name,
                                 img_path=row[img_col],
                                 annotation_path=row[annotation_col],
                                 split=row[split_col],
                                 tmp_copy=tmp_copy)


def download_batch_from_roboflow(rf_project, dataset_version: int, project_path: str, annotation_type: str,
                                 task_name: str, batch_name: str, download_date: str, platform: str):
    assert platform in ['db', 'local'], f"Platform {platform} is not supported. needs to be db or local"

    # get a list of images that in the batch
    imgs_path = paths.annotation_image_path(project=project_path,
                                            annotation_type=annotation_type,
                                            task_name=task_name,
                                            batch_name=batch_name,
                                            f_name='none.jpg')

    img_dir_name = os.path.dirname(imgs_path)

    img_list = os.listdir(img_dir_name)
    img_list = [x for x in img_list if os.path.splitext(x)[1] in ['.jpg', '.jpeg', '.tiff', '.png']]

    if annotation_type in ['object_detection', 'instance_segmentation', 'semantic_segmentation']:
        dataset = rf_project.version(dataset_version).download("coco")

        for split in ['train', 'valid', 'test']:
            dl = dataset.location + f'/{split}/_annotations.coco.json'
            data = json.load(open(dl))

            for x in range(len(data['images'])):
                anno_img_name = data['images'][x]['extra']['name']
                uid = os.path.splitext(anno_img_name)[0]

                if anno_img_name in tqdm(img_list):
                    print(f"Saving {anno_img_name} from Roboflow ...")
                    new_data = aio.extract_single_coco_json_annotations(data=data,
                                                                        index=x,
                                                                        split=split)

                    anno_path = paths.annotation_path(project=project_path,
                                                      annotation_type=annotation_type,
                                                      task_name=task_name,
                                                      batch_name=batch_name,
                                                      download_date=download_date,
                                                      f_name=uid + '.json')
                    if platform == 'db':
                        dbio.save_json_to_databricks(data=new_data,
                                                     file_name=anno_path)
                    elif platform == 'local':
                        lio.save_json(data=new_data,
                                      file_path=anno_path)
                    else:
                        raise ValueError(f"Platform {platform} is not supported.")

                else:
                    print(f"{anno_img_name} Is not in this batch, skipping saving ...")

    elif annotation_type == 'classification':
        dataset = rf_project.version(dataset_version).download("folder",
                                                               location="/tmp/roboflow_data")

        class_df = anno.generate_classification_df(folder_location=dataset.location,
                                                   img_list=img_list)

        anno_path = paths.annotation_path(project=project_path,
                                          annotation_type=annotation_type,
                                          task_name=task_name,
                                          batch_name=batch_name,
                                          download_date=download_date,
                                          f_name='classification_labels.csv')

        if platform == 'db':
            dbio.save_csv_to_databricks(data=class_df,
                                        file_name=anno_path)
        elif platform == 'local':
            class_df.to_csv(anno_path)
        else:
            raise ValueError(f"Platform {platform} is not supported.")

    else:
        raise ValueError(f"Annotation type {annotation_type} is not supported.")


def _save_annotation_job_metadata(task_name, batch_name, workspace_id, project_id, annotation_job_id):
    """ """
    annotation_job_dict = {'task_name': task_name,
                           'batch_name': batch_name,
                           'workspace_id': workspace_id,
                           'project_id': project_id,
                           'annotation_job_id': annotation_job_id}

    return annotation_job_dict


def _save_dataset_job_metadata(task_name, batch_name, workspace_id, project_id, version_id, export_format='coco'):
    """ """
    dataset_dict = {'task_name': task_name,
                    'batch_name': batch_name,
                    'workspace_id': workspace_id,
                    'project_id': project_id,
                    'version': version_id}

    return dataset_dict
