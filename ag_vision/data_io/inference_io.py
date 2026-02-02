from uuid import uuid4
from tqdm import tqdm
import os
import shutil
from ag_vision.constants import paths
from ag_vision.data_io import databricks_io as dbio
from open_aglabs.core import constants as cst


def create_inference_batch(img_list: list, project_path: str, inference_type: str, task_name: str, batch_name: str,
                           env: str = 'db'):
    assert env in ['db']
    assert inference_type in cst.ANNOTATION_TYPE_LIST, f"{inference_type} is not a valid inference type. Valid types are {cst.ANNOTATION_TYPE_LIST}"

    for img_name in tqdm(img_list):
        try:
            extension = os.path.splitext(img_name)[1]
            new_img_id = str(uuid4())

            new_img_path = paths.inference_image_path(project=project_path,
                                                      inference_type=inference_type,
                                                      task_name=task_name,
                                                      batch_name=batch_name,
                                                      f_name=new_img_id + extension)

            img_metadata_path = paths.generate_metadata_path_from_file_name(data_path=new_img_path)

            metadata = {'parent_file_path': img_name,
                        'id': new_img_id}

            if env == 'db':
                os.makedirs(os.path.dirname(new_img_path), exist_ok=True)

                shutil.copy(img_name, new_img_path)

                dbio.save_json_to_databricks(data=metadata,
                                             file_name=img_metadata_path)
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")


def save_inference_result(result: dict, project_path: str, inference_type: str, task_name: str, batch_name: str,
                          f_name: str, inference_date: str, env: str = 'db'):
    assert env in ['db']
    assert inference_type in cst.ANNOTATION_TYPE_LIST, f"{inference_type} is not a valid inference type. Valid types are {cst.ANNOTATION_TYPE_LIST}"

    result_path = paths.inference_results_path(project=project_path,
                                               inference_type=inference_type,
                                               task_name=task_name,
                                               batch_name=batch_name,
                                               inference_date=inference_date,
                                               f_name=f_name)

    if env == 'db':
        os.makedirs(os.path.dirname(result_path), exist_ok=True)

        dbio.save_json_to_databricks(data=result,
                                     file_name=result_path)