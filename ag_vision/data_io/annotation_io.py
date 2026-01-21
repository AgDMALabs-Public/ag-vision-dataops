import numpy as np
from uuid import uuid4
from tqdm import tqdm
import logging
import json
import os
import shutil
from ag_vision.constants import paths
from ag_vision.data_io import aws_io
from ag_vision.data_io import databricks_io as dbio
from open_aglabs.core import constants as cst

logger = logging.getLogger(__name__)


def create_annotation_batch(img_list: list, project_path: str, annotation_type: str, task_name: str, batch_name: str,
                            env: str = 'db'):
    assert env in ['db']
    assert annotation_type in cst.ANNOTATION_TYPE_LIST, f"{annotation_type} is not a valid annotation type. Valid types are {cst.ANNOTATION_TYPE_LIST}"

    for img_name in tqdm(img_list):
        try:
            extension = os.path.splitext(img_name)[1]
            new_img_id = str(uuid4())

            new_img_path = paths.annotation_image_path(project=project_path,
                                                       annotation_type=annotation_type,
                                                       task_name=task_name,
                                                       batch_name=batch_name,
                                                       f_name=new_img_id + extension)

            img_metadata_path = paths.generate_metadata_path_from_file_name(data_path=new_img_path)

            metadata = {'parent_img_path': img_name,
                        'parent_img_id': new_img_id}

            if env == 'db':
                os.makedirs(os.path.dirname(new_img_path), exist_ok=True)

                shutil.copy(img_name, new_img_path)

                dbio.save_json_to_databricks(data=metadata,
                                             file_name=img_metadata_path)
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")


def add_annotations_to_batch(annotations_df, project_path: str, annotation_type: str, task_name: str, batch_name: str,
                             date: str, env: str = 'db'):
    assert env in ['db']
    assert annotation_type in cst.ANNOTATION_TYPE_LIST, f"{annotation_type} is not a valid annotation type. Valid types are {cst.ANNOTATION_TYPE_LIST}"

    for col in ['image_path', 'annotation_path']:
        assert col in annotations_df.columns, f"{col} needs to be a column in annotations_df"

    for idx, row in annotations_df.iterrows():
        try:
            extension = os.path.splitext(row['image_path'])[1]
            new_img_id = str(uuid4())

            new_img_path = paths.annotation_image_path(project=project_path,
                                                       annotation_type=annotation_type,
                                                       task_name=task_name,
                                                       batch_name=batch_name,
                                                       f_name=new_img_id + extension)

            img_metadata_path = paths.generate_metadata_path_from_file_name(data_path=new_img_path)
            metadata = {'parent_img_path': row['image_path'],
                        'parent_img_id': new_img_id}

            new_annotation_path = paths.annotation_path(project=project_path,
                                                        annotation_type=annotation_type,
                                                        task_name=task_name,
                                                        batch_name=batch_name,
                                                        download_date=date,
                                                        f_name=new_img_id + '.json')

            if env == 'db':
                os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
                os.makedirs(os.path.dirname(new_annotation_path), exist_ok=True)

                # copy the image
                shutil.copy(row['image_path'], new_img_path)
                # save the new image metadata
                dbio.save_json_to_databricks(data=metadata,
                                             file_name=img_metadata_path)
                # Copy the annotation to the new dir
                shutil.copy(row['annotation_path'], new_annotation_path)
        except Exception as e:
            print(f"Error processing image {row['image_path']}: {e}")


def extract_single_coco_json_annotations(data: dict, index: int, split: str, image_name: str) -> dict:
    """
    Extract annotations and metadata for a single image from COCO dataset.

    This function extracts the annotations and relevant metadata for a single image
    from a COCO formatted JSON dataset, based on the provided index. It also allows
    updating the image's file name and specifying a dataset split for better
    organization.

    Args:
        data (dict): The COCO dataset formatted as a dictionary. It contains
            'info', 'licenses', 'categories', 'images' and 'annotations' keys.
        index (int): Index of the image in the dataset's 'images' list to retrieve
            annotations for.
        split (str): A string specifying the dataset's split (e.g., 'train',
            'val', 'test') which will be added to the 'info' section of the
            returned dictionary.
        image_name (str): A string that specifies the new file name to assign
            to the target image in the returned data.

    Returns:
        dict: A dictionary containing the extracted annotations, image metadata,
        and updated information for the single image.
    """
    # Get all the annotations that belong to that image
    annotations = [x for x in data['annotations'] if x['image_id'] == data['images'][index]['id']]

    new_data = {'info': data['info'],
                'licenses': data['licenses'],
                'categories': data['categories'],
                'images': [data['images'][index]],
                'annotations': annotations}

    new_data['info']['split'] = split
    new_data['images'][0]['file_name'] = image_name

    return new_data


def merge_coco_jsons(data_list: list[dict]):
    """
    Merges multiple single-image COCO JSON files into one unified COCO JSON file.

    Args:
        json_paths: List of paths to the individual JSON files.
        output_path: Path where the merged JSON will be saved.
    """
    merged_data = {
        "info": {},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }

    # We need to re-index images and annotations to ensure IDs are unique and sequential
    # in the merged file, although COCO IDs are strings or ints, duplicates are bad.
    # However, if your source files essentially come from the SAME dataset split,
    # the IDs might already be unique.
    # To be safe, we will trust the IDs in the files if they look like UUIDs (strings),
    # but if they are integers, we might need to offset them.
    # Based on your example, they are strings/UUIDs, so we can just aggregate them.

    seen_image_ids = set()
    seen_annotation_ids = set()
    seen_category_ids = set()

    first_file = True

    for data in data_list:
        # 1. Copy Info & Licenses from the first file (assuming they are consistent)
        if first_file:
            merged_data['info'] = data.get('info', {})
            merged_data['licenses'] = data.get('licenses', [])
            first_file = False

        # 2. Merge Categories (deduplicate by ID)
        for cat in data.get('categories', []):
            if cat['id'] not in seen_category_ids:
                merged_data['categories'].append(cat)
                seen_category_ids.add(cat['id'])

        # 3. Merge Images (deduplicate by ID)
        for img in data.get('images', []):
            if img['id'] not in seen_image_ids:
                merged_data['images'].append(img)
                seen_image_ids.add(img['id'])
            else:
                logger.warning(f"Duplicate image ID found: {img['id']}")

        # 4. Merge Annotations (deduplicate by ID)
        for ann in data.get('annotations', []):
            if ann['id'] not in seen_annotation_ids:
                merged_data['annotations'].append(ann)
                seen_annotation_ids.add(ann['id'])
            else:
                logger.warning(f"Duplicate annotation ID found: {ann['id']}")

    # Write the result
    return merged_data
