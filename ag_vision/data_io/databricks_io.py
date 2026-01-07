import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json
import tempfile
import tifffile as tif
import shutil
import cv2
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)


def upload_json_to_databricks(w: WorkspaceClient,
                              data: dict,
                              file_name: str,
                              overwrite: bool = False):

    json_content = json.dumps(data, indent=4).encode('utf-8')

    with w.dbfs.open(file_name, write=True, overwrite=False) as f:
        f.write(json_content)


def upload_file_with_progress(w, local_file_path, volume_path, chunk_size=1024*1024):
    """
    Uploads a file to Databricks Volumes with a progress bar.
    """
    file_size = os.path.getsize(local_file_path)

    # Open the file and wrap it with tqdm.wrapattr
    # This creates a file-like object that updates the progress bar on read()
    with open(local_file_path, "rb") as f:
        with tqdm.wrapattr(f, "read", total=file_size, unit="B", unit_scale=True, desc="Uploading") as f_wrapped:
            w.files.upload(volume_path, f_wrapped, overwrite=True)

    logger.info(f"Upload complete: {volume_path}")


def upload_file_with_progress_1(
        w: WorkspaceClient,
        local_file_path: str,
        volume_path: str,
        chunk_size: int = 1024 * 1024 * 10):
    """
    Uploads a file to a specified volume path with progress tracking. Displays a progress
    bar indicating upload progress and updates it as chunks of the file are read and
    uploaded. This function ensures that the file upload is performed in chunks, optimizing
    memory usage and providing detailed progress updates.

    Arguments:
        w: WorkspaceClient
            An instance of WorkspaceClient used to upload the file.
        local_file_path: str
            The path of the local file to be uploaded.
        volume_path: str
            The path in the target volume where the file will be uploaded.
        chunk_size: int, optional
            The size of each chunk to be uploaded, in bytes. Defaults to 1 MB.

    Returns:
        None
    """
    file_size = os.path.getsize(local_file_path)

    with (
        open(local_file_path, "rb") as f,
        tqdm(total=file_size, unit="B", unit_scale=True, desc="Uploading") as pbar,
    ):

        def stream():
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                pbar.update(len(chunk))
                yield chunk

        w.files.upload(volume_path, stream(), overwrite=True)
        logger.info(f"Upload complete: {volume_path}")


def save_tif_to_databricks(img: np.ndarray, file_name: str):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_f:
            tif.imwrite(temp_f.name, img)
            temp_f_path = temp_f.name

        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        shutil.move(temp_f_path, file_name)
    except Exception as e:
        print(f"Failed to save image from databricks: {e}")
        return None


def read_tif_from_databricks(file_name: str):
    try:
        with tempfile.NamedTemporaryFile(suffix=".tif") as temp_f:
            temp_f_path = temp_f.name

            os.makedirs(os.path.dirname(temp_f_path), exist_ok=True)
            shutil.copy(file_name, temp_f_path)
            return tif.imread(temp_f_path)
    except Exception as e:
        print(f"Failed to read image from databricks: {e}")
        return None


def read_img_from_databricks(file_name: str):
    try:
        if not os.path.exists(file_name):
            print(f"File not found: {file_name}")
            return None

        # cv2.imread works directly on Volume paths
        img = cv2.imread(file_name, cv2.IMREAD_COLOR)

        if img is None:
            print("Failed to decode image.")

        return img
    except Exception as e:
        print(f"Failed to read image from databricks: {e}")
        return None


def save_img_to_databricks(img: np.ndarray, file_name: str):
    try:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        success = cv2.imwrite(file_name, img)

        return success
    except Exception as e:
        print(f"Failed to save image from databricks: {e}")


def read_json_from_databricks(file_name: str):
    try:
        with open(file_name, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to read json from databricks: {e}")


def save_json_to_databricks(data: dict, file_name: str):
    try:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        json_str = json.dumps(data, indent=4)
        with open(file_name, 'w') as f:
            f.write(json_str)

    except Exception as e:
        print(f"Failed to save json to databricks: {e}")


def read_csv_from_databricks(file_name: str) -> pd.DataFrame:
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv") as temp_f:
            temp_f_path = temp_f.name

            os.makedirs(os.path.dirname(temp_f_path), exist_ok=True)
            shutil.copy(file_name, temp_f_path)

            return pd.read_csv(file_name)
    except Exception as e:
        print(f"Failed to save json from databricks: {e}")


def save_csv_to_databricks(data: pd.DataFrame, file_name: str):
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv") as temp_f:
            data.to_csv(temp_f.name, index=False)
            temp_f_path = temp_f.name

            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            shutil.move(temp_f_path, file_name)
    except Exception as e:
        print(f"Failed to save json from databricks: {e}")
