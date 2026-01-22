import os
import shutil
import json

from tqdm import tqdm
from ag_vision.constants import paths
from ag_vision.data_io import annotation_io as aio
from ag_vision.data_io import databricks_io as dbio
from ag_vision.data_io import local_io as lio
from ag_vision.annotation import annotation as anno
import logging
import tempfile

logger = logging.getLogger(__name__)

SPLIT_LIST = ['train', 'valid', 'test']
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.tiff', '.png', '.webp']


def upload_image_to_roboflow(rf_project, batch_name: str, img_path: str, annotation_path: str = None,
                             split: str = 'train', tmp_copy: bool = True):
    if tmp_copy:
        tmp_img_path = f"{tempfile.mktemp()}/images/{os.path.basename(img_path)}"
        os.makedirs(os.path.dirname(tmp_img_path), exist_ok=True)
        shutil.copy(img_path, tmp_img_path)

        if annotation_path is not None:
            tmp_annotation_path = f"{tempfile.mktemp()}/annotations/{os.path.basename(annotation_path)}"
            os.makedirs(os.path.dirname(tmp_annotation_path), exist_ok=True)
            shutil.copy(annotation_path, tmp_annotation_path)
        else:
            tmp_annotation_path = None
    else:
        tmp_img_path = img_path
        tmp_annotation_path = annotation_path

    rf_project.upload(
        image_path=tmp_img_path,
        annotation_path=tmp_annotation_path,
        split=split,  # Optional: "train", "valid", or "test"
        batch_name=batch_name
    )


def upload_annotation_batch_to_roboflow(rf_project, annotation_type: str, project_path: str, task_name: str,
                                        batch_name: str, download_date: str, split: str, tmp_copy: bool = True,
                                        img_extension: list = IMG_EXTENSIONS,
                                        annotation: bool = True):
    """
    Uploads a batch of images and their annotations to Roboflow. The function processes
    a given directory of images and optionally their associated annotations in COCO JSON
    format. Images and annotations are prepared, merged if required, and then uploaded
    to a Roboflow project.

    Args:
        rf_project: The name of the Roboflow project where the images and annotations
            will be uploaded.
        annotation_type: The type of annotation format used in the project
            (e.g., bounding boxes, segmentation).
        project_path: The path to the project directory containing images and associated
            annotations.
        task_name: The name of the task related to the given batch of images
            and annotations.
        batch_name: The name of the batch of images and annotations being processed.
        download_date: The download date of the annotation files to be processed
            and uploaded.
        split: The data split name used, such as train, validation, or test.
        tmp_copy: Boolean flag indicating whether or not to temporarily copy/prepare
            files before uploading. Defaults to True.
        img_extension: A list of valid image file extensions supported by the function
            during processing. Defaults to ['.jpg', '.jpeg', '.tiff', '.png'].
        annotation: Boolean flag indicating whether annotation files should be processed
            and uploaded along with images. Defaults to True.
    """
    # Get the dir name with all the images.
    imgs_path = paths.annotation_image_path(project=project_path,
                                            annotation_type=annotation_type,
                                            task_name=task_name,
                                            batch_name=batch_name,
                                            f_name='none.jpg')

    img_dir = os.path.dirname(imgs_path)

    if download_date is not None:
        annotation_path = paths.annotation_path(project=project_path,
                                                annotation_type=annotation_type,
                                                task_name=task_name,
                                                batch_name=batch_name,
                                                download_date=download_date,
                                                f_name='none.jpg')
        annotation_dir = os.path.dirname(annotation_path)
    else:
        annotation_dir = '/'

    img_files = os.listdir(img_dir)
    img_files = [x for x in img_files if '.json' not in x]
    imgs = []

    for f_name in img_files:
        if os.path.splitext(f_name)[1] in img_extension:
            imgs.append(f_name)
        else:
            print(f'Skipping {f_name} as its file type is not supported.')

    annotation_data_list = []
    if annotation:
        print("Generating Merged Annotation File...")
        for img in tqdm(imgs):
            # Assumes that the annotation is the same name but ends in .json
            a_file = annotation_dir + '/' + os.path.splitext(img)[0] + '.json'

            try:
                # assumes COCO Json format.
                a = dbio.read_json_from_databricks(file_name=a_file)
                for image_entry in a['images']:
                    image_entry['file_name'] = img

                annotation_data_list.append(a)
            except Exception as e:
                print(f"Error reading annotation file {a_file}: {e}")
                continue

        print("Merging annotations ...")
        merged_data = aio.merge_coco_jsons(data_list=annotation_data_list)
        merged_file = annotation_dir + '/' + 'merged.json'
        dbio.save_json_to_databricks(data=merged_data,
                                     file_name=merged_file)

        print("Uploading Images and Annotations ...")
        for img in tqdm(imgs):
            try:
                print(f"Uploading {img} and annotation to Roboflow")
                upload_image_to_roboflow(rf_project=rf_project,
                                         batch_name=batch_name,
                                         img_path=img_dir + '/' + img,
                                         annotation_path=merged_file,
                                         split=split,
                                         tmp_copy=tmp_copy)
            except Exception as e:
                print(f"Error uploading {img}: {e}")

    else:
        for img in tqdm(imgs):
            try:
                print(f"Uploading {img} to Roboflow")
                upload_image_to_roboflow(rf_project=rf_project,
                                         batch_name=batch_name,
                                         img_path=img_dir + '/' + img,
                                         annotation_path=None,
                                         split=split,
                                         tmp_copy=tmp_copy)
            except Exception as e:
                print(f"Error uploading {img}: {e}")


def upload_image_batch_to_roboflow(rf_project, annotation_type: str, project_path: str, task_name: str,
                                   batch_name: str, split: str, tmp_copy: bool = True,
                                   img_extension: list = IMG_EXTENSIONS):
    upload_annotation_batch_to_roboflow(rf_project=rf_project,
                                        annotation_type=annotation_type,
                                        project_path=project_path,
                                        task_name=task_name,
                                        batch_name=batch_name,
                                        download_date=None,
                                        split=split,
                                        tmp_copy=tmp_copy,
                                        img_extension=img_extension,
                                        annotation=False)


def _save_image_from_annotation_download(download_dir: str, save_dir: str):
    for split in SPLIT_LIST:
        split_dir = download_dir + '/' + split
        if os.path.exists(split_dir):
            image_list = os.listdir(split_dir)
            image_list = [x for x in image_list if '.json' not in x]

            for img_name in tqdm(image_list):
                save_path = save_dir + '/' + img_name.replace('.rf.', '_rf_')
                if os.path.exists(save_path):
                    print('The image already exists, skipping download.')
                    continue
                else:
                    shutil.copy(download_dir + '/' + split + '/' + img_name, save_path)


def _save_image_from_classifiction_download(download_dir: str, save_dir: str):
    for split in SPLIT_LIST:
        split_dir = download_dir + '/' + split
        if os.path.exists(split_dir):
            classes = os.listdir(split_dir)

            for cls in classes:
                class_dir = split_dir + '/' + cls

                image_list = os.listdir(class_dir)
                image_list = [x for x in image_list if '.json' not in x]

                for img_name in tqdm(image_list):
                    save_path = save_dir + '/' + img_name.replace('.rf.', '_rf_')
                    if os.path.exists(save_path):
                        print('The image already exists, skipping download.')
                        continue
                    else:
                        shutil.copy(download_dir + '/' + split + '/' + cls + '/' + img_name, save_path)


def download_annotation_batch_from_roboflow(rf_project, dataset_version: int, project_path: str, annotation_type: str,
                                            task_name: str, batch_name: str, download_date: str, platform: str,
                                            save_images: bool = False):
    assert platform in ['db', 'local'], f"Platform {platform} is not supported. needs to be db or local"

    # get a list of images that in the batch
    imgs_path = paths.annotation_image_path(project=project_path,
                                            annotation_type=annotation_type,
                                            task_name=task_name,
                                            batch_name=batch_name,
                                            f_name='none.jpg')

    img_dir_name = os.path.dirname(imgs_path)
    if not os.path.exists(img_dir_name):
        os.makedirs(img_dir_name, exist_ok=True)

    if annotation_type in ['object_detection', 'instance_segmentation', 'semantic_segmentation']:
        dataset = rf_project.version(dataset_version).download("coco")

        if save_images:
            _save_image_from_annotation_download(download_dir=dataset.location,
                                                 save_dir=img_dir_name)

        # get a list of the images in the dir
        img_list = os.listdir(img_dir_name)
        img_list = [x for x in img_list if os.path.splitext(x)[1] in IMG_EXTENSIONS]

        for split in SPLIT_LIST:
            split_file = dataset.location + f'/{split}/_annotations.coco.json'
            if os.path.exists(split_file):
                data = json.load(open(split_file))

                for x in range(len(data['images'])):
                    if save_images:
                        anno_img_name = data['images'][x]['file_name'].replace('.rf.', '_rf_')
                    else:
                        anno_img_name = data['images'][x]['extra']['name']

                    uid = os.path.splitext(anno_img_name)[0]

                    if anno_img_name in img_list:
                        print(f"Saving {anno_img_name} from Roboflow ...")
                        new_data = aio.extract_single_coco_json_annotations(data=data,
                                                                            index=x,
                                                                            split=split,
                                                                            image_name=anno_img_name)

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
                                                               location=f"{tempfile.mktemp()}/roboflow_data")

        if save_images:
            _save_image_from_classifiction_download(download_dir=dataset.location,
                                                    save_dir=img_dir_name)

        # get a list of the images in the dir
        img_list = os.listdir(img_dir_name)
        img_list = [x for x in img_list if os.path.splitext(x)[1] in IMG_EXTENSIONS]
        print(f"Number of images in the batch: {len(img_list)}")

        class_df = anno.generate_classification_df(folder_location=dataset.location,
                                                   img_list=img_list,
                                                   downloaded_images=save_images)

        print(f"Number of annotations in the batch: {len(class_df)}")

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
