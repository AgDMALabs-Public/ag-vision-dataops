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


def save_annotated_image_and_metadata_to_s3(image: np.ndarray, metadata: dict, project_path: str, annotation_type: str,
                                            task_name: str, batch_name: str, image_name: str, bucket_name):
    img_path = paths.annotation_image_path(project=project_path,
                                           annotation_type=annotation_type,
                                           task_name=task_name,
                                           batch_name=batch_name,
                                           f_name=image_name)

    metadata_path = paths.generate_metadata_path_from_file_name(img_path)

    aws_io.save_image_to_s3(image=image,
                            bucket_name=bucket_name,
                            key=img_path)

    aws_io.save_json_to_s3(json_data=json.dumps(metadata, indent=4),
                           bucket_name=bucket_name,
                           key=metadata_path)


def create_annotation_batch(img_list: list, project_path: str, annotation_type: str, task_name: str, batch_name: str,
                            env: str = 'db'):
    assert env in ['db']
    assert annotation_type in cst.ANNOTATION_TYPE_LIST, f"{annotation_type} is not a valid annotation type. Valid types are {cst.ANNOTATION_TYPE_LIST}"

    for img_name in tqdm(img_list):
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


def extract_single_coco_json_annotations(data: dict, index: int, split: str) -> dict:
    assert data['images'][index]['id'] == data['annotations'][index]['image_id']

    new_data = {'info': data['info'],
                'licenses': data['licenses'],
                'categories': data['categories'],
                'images': [data['images'][index]],
                'annotations': [data['annotations'][index]]}

    new_data['info']['split'] = split
    new_data['images'][0]['file_name'] = new_data['images'][0]['extra']['name']

    return new_data
