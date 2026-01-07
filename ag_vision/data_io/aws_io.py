import json
import numpy as np
import boto3
from botocore.exceptions import ClientError
import logging
import cv2
from PIL import Image, ExifTags
from io import BytesIO

logger = logging.getLogger(__name__) 


def save_json_to_s3(json_data: str, bucket_name: str, key: str):
    """
    Uploads a dictionary to an Amazon S3 bucket as a JSON file.

    Parameters:
    json_data (str): The json string to save to AWS
    bucket_name (str): The name of the target S3 bucket.
    key (str): The target file name for the JSON file in the S3 bucket.

    This function serializes a given dictionary into JSON format and uploads it
    to the specified S3 bucket with the provided file name. If any error occurs
    during the process, it prints an error message.
    """
    try:
        # Initialize the S3 client
        s3 = boto3.client('s3')

        # Upload the JSON string to S3
        s3.put_object(Bucket=bucket_name,
                      Key=key,
                      Body=json_data.encode('utf-8'),
                      ContentType='application/json')

        logger.info(f"Dictionary successfully saved to 's3://{bucket_name}/{key}'")

    except Exception as e:
        logger.error(f"An error occurred while saving to S3: {e}")


def save_image_to_s3(image, bucket_name: str, key: str):
    """
    Saves an image to an AWS S3 bucket.

    :param image: The image to save (should be a NumPy array, compatible with OpenCV).
    :param bucket_name: The name of the S3 bucket where the image will be saved.
    :param key: The S3 key (file path) to save the image.
    :param aws_region: AWS region where the S3 bucket is located (default: "us-east-1").
    :return: True if the image is uploaded successfully, else False.
    """
    try:
        # Initialize the S3 client
        s3_client = boto3.client("s3")

        # Encode the OpenCV image (NumPy array) to a binary format such as JPEG
        success, encoded_image = cv2.imencode('.jpg', image)
        if not success:
            logger.warning("Error: Failed to encode the image to a binary format.")
            return False

        # Convert the binary image to bytes
        img_bytes = BytesIO(encoded_image.tobytes())

        # Upload the image to S3
        s3_client.upload_fileobj(
            img_bytes,  # The image in binary (file-like object)
            Bucket=bucket_name,  # Target S3 bucket
            Key=key  # S3 object path (key)
        )

        logger.warning(f"Image successfully uploaded to s3://{bucket_name}/{key}")
        return True

    except Exception as e:
        logger.warning(f"Exception occurred while saving image to S3: {e}")
        return False


def upload_file_to_s3(file_path: str, bucket_name: str, key: str) -> None:
    """
    Uploads a file to an Amazon S3 bucket.

    file_path: The local file path of the file to be uploaded.
    bucket_name: The name of the target S3 bucket.
    object_name: The name of the file in the S3 bucket.

    Returns True if the file was successfully uploaded.
    Logs an error and handles exceptions in case of a ClientError, returning None.
    """

    # Create an S3 client
    s3_client = boto3.client('s3')

    try:
        # Upload the file
        response = s3_client.upload_file(file_path,
                                         bucket_name,
                                         key)

        logging.info(f"Successfully uploaded '{key}' to '{bucket_name}, with a response of {response}'")

    except ClientError as e:
        logging.error(f"Error, {e}")


def list_s3_objects(bucket_name: str, s3_prefix: str) -> list[str]:
    """
    Lists files in an S3 bucket under a specified prefix.

    Parameters:
    bucket_name: The name of the S3 bucket.
    s3_prefix: The key prefix within the S3 bucket to filter the listed files.

    Returns:
    A list of strings, where each string is the key of an object in the bucket.

    Exceptions:
    Logs the error if the bucket does not exist or if access is denied.
    Handles AWS client errors, unexpected exceptions, and returns an empty list in case of error.
    """
    s3 = boto3.client('s3')
    file_list = []
    try:
        # Use a paginator to handle buckets with a large number of objects
        # The list_objects_v2 API call can return up to 1000 keys at a time.
        paginator = s3.get_paginator('list_objects_v2')
        operation_params = {'Bucket': bucket_name, 'Prefix': s3_prefix}

        pages = paginator.paginate(**operation_params)

        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    file_list.append(obj["Key"])
        logger.info(f"Successfully listed files in bucket: {bucket_name}")
        return file_list
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == "NoSuchBucket":
            logger.warning(f"Error: The bucket '{bucket_name}' does not exist.")
        elif error_code == "AccessDenied":
            logger.warning(f"Error: Access denied to bucket '{bucket_name}'. "
                           f"Please check your AWS credentials and bucket permissions.")
        else:
            logger.warning(f"An AWS client error occurred: {e}")
        return []
    except Exception as e:
        logger.warning(f"An unexpected error occurred: {e}")
        return []


def read_image_from_s3(bucket_name: str, key: str) -> np.ndarray or None:
    """
    Reads an image from an AWS S3 bucket and returns it in a format suitable for OpenCV.

    :param bucket_name: The name of the S3 bucket.
    :param key: The key (path) of the image in the S3 bucket.
    :return: Image as a NumPy array suitable for OpenCV.
    """
    # Initialize an S3 client
    s3_client = boto3.client("s3")

    try:
        # Download the image as an in-memory file
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        image_data = response["Body"].read()

        # Convert the image data to a NumPy array (compatible with OpenCV)
        image_array = np.frombuffer(image_data, np.uint8)

        # Decode the image using OpenCV
        image_cv = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        return image_cv
    except Exception as e:
        logger.warning(f"Error reading image from S3: {e}")
        return None


def read_exif_from_s3(bucket_name: str, key: str) -> dict:
    """
    Retrieves EXIF metadata from an image stored in AWS S3.

    :param bucket_name: str. Name of the AWS S3 bucket.
    :param key: str. Key (path) of the image in the S3 bucket.
    :return: dict. Dictionary containing EXIF metadata.
    """
    # Initialize S3 client
    s3 = boto3.client('s3')
    exif_data = {}

    try:
        # Download the image file into a stream (BytesIO buffer)
        file_obj = s3.get_object(Bucket=bucket_name, Key=key)
        file_stream = BytesIO(file_obj['Body'].read())

        # Open the image with Pillow
        with Image.open(file_stream) as img:
            # Get raw EXIF data
            raw_exif = img._getexif()

            if raw_exif is not None:
                # Convert raw EXIF data into human-readable tags
                for tag_id, value in raw_exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, tag_id)  # Get readable tag name
                    exif_data[tag_name] = value
            else:
                print("No EXIF metadata found in the image.")

    except Exception as e:
        print(f"Error retrieving or processing the image: {e}")

    return exif_data


def read_json_from_s3(bucket_name: str, object_key: str) -> dict or None:
    """
    Reads a JSON file from an S3 bucket and returns its contents as a dictionary.

    :param bucket_name: str. The name of the S3 bucket.
    :param object_key: str. The key (path) of the JSON file in the S3 bucket.
    :return: dict. The parsed JSON data as a Python dictionary, or None if an error occurs.
    """
    # Create an S3 client
    s3 = boto3.client('s3')

    try:
        # Fetch the JSON file from S3
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        data = response['Body'].read().decode('utf-8')  # Read and decode the file content

        # Parse the JSON content into a dictionary
        json_data = json.loads(data)
        logger.info("Successfully read JSON data from S3.")
        return json_data

    except boto3.exceptions.S3UploadFailedError as e:
        logger.warning(f"Error retrieving file from S3: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"Error decoding JSON data: {e}")
        return None
    except Exception as e:
        logger.warning(f"An unexpected error occurred: {e}")
        return None


def download_file_from_s3(bucket_name: str, object_key: str, download_path: str) -> str or None:
    """
    Downloads an image from an S3 bucket.

    :param bucket_name: str. The name of the S3 bucket.
    :param object_key: str. The key (path) of the object in the S3 bucket.
    :param download_path: str. The local path where the image should be saved.
    :return: str. The local download path of the image.
    """

    # Create an S3 client
    s3 = boto3.client('s3')

    try:
        # Download the file from S3
        s3.download_file(Bucket=bucket_name, Key=object_key, Filename=download_path)
        print(f"Image successfully downloaded to {download_path}")
        return download_path

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
