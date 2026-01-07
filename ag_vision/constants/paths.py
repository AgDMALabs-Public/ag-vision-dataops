import logging
import pandas as pd
from open_aglabs.core.constants import CROP_LIST, YEAR_LIST, COUNTRY_CODES, TIME_OF_YEAR_LIST, ANNOTATION_TYPE_LIST
import os

logger = logging.getLogger(__name__)  # Use __name__ to get the module's name


# ----------------------------------------------------------Helper------------------------------------------------------#
def _format_date_string(date: str) -> str:
    """
    Formats a given datetime string into a specific date format (YYYY-MM-DD).

    Parameters:
    date (str): A string representation of a datetime to be formatted.

    Returns:
    str: The formatted date string in the format YYYY-MM-DD.
    """
    date_object = pd.to_datetime(date)
    date_object = date_object.strftime("%Y-%m-%d")

    return date_object


def _format_datetime_string(datetime: str) -> str:
    """
    Converts a datetime string into a specified datetime format.

    Args:
        datetime (str): The datetime string to be formatted.

    Returns:
        str: The formatted datetime string in the format "YYYY-MM-DD HH:MM:SS".
    """
    date_object = pd.to_datetime(datetime)
    date_object = date_object.strftime("%Y-%m-%d %H:%M:%S")

    return date_object


def generate_metadata_path_from_file_name(data_path: str) -> str:
    """
    Generates a metadata file path from a given file name.

    This function takes a file path, extracts its extension, and replaces it
    with '.json' to generate the corresponding metadata file path.

    Args:
        data_path: The original file path as a string.

    Returns:
        A string representing the metadata file path with the '.json' extension.
    """
    extension = os.path.splitext(data_path)[1]
    return data_path.replace(extension, '.json')


# ---------------------------------------------Trial / Field / Base Paths-----------------------------------------------#
def season_code(year: int, country: str, crop: str, time_of_year: str) -> str:
    """
    Generates a season identifier string based on the provided year, country, crop, and time of year.

    Args:
        year (int): The year for the season; must be in the predefined YEAR_LIST.
        country (str): The country code; must be one of the keys in the COUNTRY_CODES dictionary.
        crop (str): The crop name; must be in the predefined CROP_LIST.
        time_of_year (str): The time of year; must be in the predefined TIME_OF_YEAR_LIST.

    Returns:
        str: A formatted string containing the year, country code, crop, and time of year in the
        format "year:country:crop:time_of_year".

    Raises:
        AssertionError: If the year is not in YEAR_LIST.
        AssertionError: If the country is not a valid key in COUNTRY_CODES.
        AssertionError: If the crop is not in CROP_LIST.
        AssertionError: If the time_of_year is not in TIME_OF_YEAR_LIST.
    """
    country = country.upper()
    crop = crop.lower()
    time_of_year = str(time_of_year).lower()

    assert year in YEAR_LIST, f'The years allowed are {YEAR_LIST}'
    assert country in COUNTRY_CODES.keys(), f'{country} The country codes allowed are {COUNTRY_CODES}'
    assert crop in CROP_LIST, f'{crop} The crops allowed are {CROP_LIST}'
    assert time_of_year in TIME_OF_YEAR_LIST, f'The times of year allowed are {TIME_OF_YEAR_LIST}'

    season = f"{year}:{country}:{crop}:{time_of_year}"
    season = season.lower()
    logger.debug(f"The season code is {season}")

    return season


def location_path(project: str, site: str, trial: str, season: str, field: str, location: str) -> str:
    """
    Generates a location path by combining various input parameters, converting them to lowercase, and structuring them
    in a specific order. A location is defined in terms of BrApi and has a unique set of ranges and columns for a single
    plot map.

    Parameters:
    site (str): The name of the site. It will be converted to lowercase.
    trial (str): The name of the trial. It will be converted to lowercase.
    season (str): The name of the season. This string is included as is.
    field (str): The name of the field. It will be converted to lowercase.
    location (str): The name of the location. It will be converted to lowercase.

    Returns:
    str: A formatted string representing the location path in the format "site/trial/season/field/location/" with the site, trial, field, and location strings in lowercase.
    """
    site = site.lower()
    trial = trial.lower()
    field = field.lower()
    location = location.lower()

    loc_path = f"{project}/{site}/{trial}/{season}/{field}/{location}"
    logger.debug(f"The location path is {loc_path}")

    return loc_path


# ------------------------------------------------------Image Paths-----------------------------------------------------#
def trial_image_path(location_path: str, task: str, protocol: str, date: str, plot_id: str,
                     image_name: str) -> str:
    """
    Generates the file path for a trial image based on the given parameters.

    Parameters:
    location_path (str): The base path for the field study directory.
    task (str): The task name associated with the trial image.
    protocol (str): The protocol name used during the trial.
    date (str): The date associated with the day the image was collected, to be formatted into a specific date format.
    plot_id (str): The identifier for the plot related to the trial.
    image_name (str): The name of the image file.

    Returns:
    str: A formatted file path string for the trial image.
    """
    task = task.lower()
    protocol = protocol.lower()
    plot_id = plot_id.lower()
    image_name = image_name.lower()

    date = _format_date_string(date)

    return f"{location_path}/{task}/images/{protocol}/{date}/{plot_id}/{image_name}"


def scouting_image_path(location_path: str, task: str, protocol: str, date: str,
                        image_name: str) -> str:
    """
    Generates a file path for a scouting image based on the provided parameters.

    Parameters:
    location_path (str): The base path for the field study directory.
    task (str): The task name to be included in the path. This will be converted to lowercase.
    protocol (str): The protocol name to be included in the path. This will be converted to lowercase.
    date (str): A string representing the date, which will be formatted for inclusion in the path.
    image_name (str): The name of the image file. This will be converted to lowercase.

    Returns:
    str: A formatted file path string for the scouting image.
    """
    task = task.lower()
    protocol = protocol.lower()
    image_name = image_name.lower()

    date = _format_date_string(date)

    return f"{location_path}/{task}/images/{protocol}/{date}/{image_name}"


# ----------------------------------------------Annotations------------------------------------------------------------#
def annotation_image_path(project: str, annotation_type: str, task_name: str, batch_name: str, f_name: str) -> str:
    """
    Generates the file path for an annotation image based on the specified parameters.

    Parameters:
    annotation_type (str): The type of annotation. Must be a valid type present in ANNOTATION_TYPE_LIST.
    task_name (str): The name of the associated task.
    batch_name (str): The name of the batch associated with the annotation.
    f_name (str): The file name of the image.

    Returns:
    str: A string representing the generated file path for the annotation image.

    Raises:
    AssertionError: If the provided annotation_type is not in ANNOTATION_TYPE_LIST.
    """
    annotation_type = annotation_type.lower()
    assert annotation_type.lower() in ANNOTATION_TYPE_LIST, f"Invalid annotation type: '{annotation_type}'. Available options are: {ANNOTATION_TYPE_LIST}"

    task_name = task_name.lower()
    batch_name = batch_name.lower()
    f_name = f_name.lower()

    return f"{project}/annotations/{annotation_type}/{task_name}/{batch_name}/images/{f_name}"


def annotation_path(project: str, annotation_type: str, task_name: str, batch_name: str, download_date: str,
                    f_name: str) -> str:
    """
    Generates a file path for annotations based on the provided parameters.

    Args:
        annotation_type (str): The type of annotation (e.g., 'object_detection', 'instance_segmentation', 'classification).
        task_name (str): The name of the task associated with the annotation.
        batch_name (str): The name of the batch for the task.
        download_date (str): The date when the annotations were downloaded. Formatted using _format_date_string.
        f_name (str): The name of the file.

    Returns:
        str: A formatted string representing the path for the annotation in the form:
             "{project}/annotations/{annotation_type}/{task_name}/{batch_name}/{download_date}/{f_name}"
    """
    annotation_type = annotation_type.lower()
    assert annotation_type.lower() in ANNOTATION_TYPE_LIST, f"Invalid annotation type: '{annotation_type}'. Available options are: {ANNOTATION_TYPE_LIST}"

    task_name = task_name.lower()
    batch_name = batch_name.lower()
    f_name = f_name.lower()

    download_date = _format_date_string(download_date)

    return f"{project}/annotations/{annotation_type}/{task_name}/{batch_name}/{download_date}/{f_name}"


# ---------------------------------------------------Models------------------------------------------------------------#
def model_weight_path(project: str, model_name: str, version: str, f_name: str) -> str:
    """
    Generates a structured file path for model weights based on provided project, model name, version, and file name.
    The model name, version, and file name are automatically converted to lower case during the path generation.

    Args:
        project (str): The name of the project.
        model_name (str): The name of the model.
        version (str): The version of the model.
        f_name (str): The name of the weight file.

    Returns:
        str: The generated file path for the model weight.
    """
    model_name = model_name.lower()
    version = version.lower()
    f_name = f_name.lower()

    return f"{project}/models/{model_name}/{version}/weights/{f_name}"


def model_dataset_dir(project: str, model_name: str, version: str) -> str:
    """
    Generates the directory path for a specified model dataset.

    This function constructs a standardized directory path for a model's dataset
    based on the provided project name, model name, and version. It ensures the
    model name and version are transformed to lowercase for consistency.

    Args:
        project: The name of the project as a string.
        model_name: The name of the model as a string.
        version: The version of the model as a string.

    Returns:
        The directory path as a string.
    """
    model_name = model_name.lower()
    version = version.lower()

    return f"{project}/models/{model_name}/{version}/dataset"


# ---------------------------------------------------field Notes-------------------------------------------------------#
def trial_note_path(location_path: str, task: str, protocol: str, date: str, plot_id: str, note_name: str) -> str:
    task = task.lower()
    protocol = protocol.lower()
    plot_id = plot_id.lower()
    note_name = note_name.lower()

    date = _format_date_string(date)

    return f"{location_path}/{task}/notes/{protocol}/{date}/{plot_id}/{note_name}"


def scouting_note_path(location_path: str, task: str, protocol: str, date: str, note_name: str) -> str:
    task = task.lower()
    protocol = protocol.lower()
    note_name = note_name.lower()

    date = _format_date_string(date)

    return f"{location_path}/{task}/notes/{protocol}/{date}/{note_name}"


# ---------------------------------------------------Survey Data-------------------------------------------------------#
def survey_path(location_path: str, task: str, protocol: str, date: str, survey_obj_name: str) -> str:
    task = task.lower()
    protocol = protocol.lower()
    survey_obj_name = survey_obj_name.lower()

    date = _format_date_string(date)

    return f"{location_path}/{task}/survey/{protocol}/{date}/{survey_obj_name}"


# ---------------------------------------------------Drones------------------------------------------------------------#
def drone_mission_dir(location_path: str, mission_name: str):
    """
    Generates a formatted directory path string for a drone mission.

    This function constructs a string that represents the directory path for
    a specific drone mission. It uses the provided location path, date, and
    mission name to create the formatted result. The mission name is
    transformed to lowercase before being added to the path.

    Parameters:
    location_path: str
        The base location directory path where the mission will be stored.

    mission_name: str
        The name of the mission. This is transformed to lowercase for
        consistency in the output path.

    Returns:
    str
        A formatted string representing the full path for the drone mission.
    """
    mission_name = mission_name.lower()


    return f"{location_path}/drone/{mission_name}"


def drone_study_boundary_path(drone_mission_dir: str) -> str:
    """
    Constructs the full file path for the study boundary GeoJSON of a given task.

    Parameters:
    location_path: str
        The base file path of the field study.

    Returns:
    str
        The complete file path to the study boundary GeoJSON.
    """
    drone_mission_dir = os.path.dirname(drone_mission_dir)
    return f"{drone_mission_dir}/field_data/study_boundary.geojson"


def drone_plot_boundary_path(drone_mission_dir: str) -> str:
    """
    Generates the file path for a plot boundary GeoJSON file.

    This function constructs a file path string for a GeoJSON file representing
    the plot boundary of a study field based on the given field study path and
    task name.

    Parameters:
    location_path: str
        The base path to the field study directory.
    task: str
        The task name associated with the study.

    Returns:
    str
        The full file path to the plot boundary GeoJSON file.
    """
    return f"{drone_mission_dir}/field_data/plot_boundary.geojson"


def drone_mission_ground_control_point_path(drone_mission_dir: str) -> str:
    """
    Generates the file path for storing the flight details of a drone.

   Parameters:
    location_path: str
        The base path to the field study directory.

    Returns:
    str: The complete file path for the flight details JSON file.
    """
    return f"{drone_mission_dir}/field_data/ground_control_points.geojson"


def drone_flight_details_path(drone_mission_dir: str, flight_date: str) -> str:
    """
    Generates the file path for storing the flight details of a drone.

    Parameters:
    drone_mission_dir (str): The base directory path where drone flight data is stored.

    Returns:
    str: The complete file path for the flight details JSON file.
    """
    flight_date = _format_date_string(flight_date)
    return f"{drone_mission_dir}/{flight_date}/flight_details.json"


def drone_raw_flight_data(drone_mission_dir: str, flight_date: str, file_name: str) -> str:
    """
    Constructs the full file path for a specific image from a drone flight.

    Args:
    drone_mission_dir (str): The base directory path for the drone flight.
    image_name (str): The name of the image file.

    Returns:
    str: The complete file path to the specific image.
    """
    flight_date = _format_date_string(flight_date)
    file_name = file_name.lower()

    return f"{drone_mission_dir}/{flight_date}/raw_data/{file_name}"


def drone_flight_orthomosaic_path(drone_mission_dir: str, flight_date: str, method: str, ortho_date: str, image_name: str) -> str:
    """
    Constructs the full file path for a specific image from a drone flight.

    Args:
    drone_mission_dir (str): The base directory path for the drone flight.
    image_name (str): The name of the image file.

    Returns:
    str: The complete file path to the specific image.
    """
    flight_date = _format_date_string(flight_date)
    ortho_date = _format_date_string(ortho_date)
    image_name = image_name.lower()

    return f"{drone_mission_dir}/{flight_date}/orthomosaic/{method}_{ortho_date}/{image_name}"


def drone_flight_dem_path(drone_mission_dir: str, flight_date: str, method: str, dem_date: str, image_name: str) -> str:
    """
    Constructs the full file path for a specific image from a drone flight.

    Args:
    drone_mission_dir (str): The base directory path for the drone flight.
    image_name (str): The name of the image file.

    Returns:
    str: The complete file path to the specific image.
    """
    dem_date = _format_date_string(dem_date)
    flight_date = _format_date_string(flight_date)
    image_name = image_name.lower()

    return f"{drone_mission_dir}/{flight_date}/dem/{method}_{dem_date}/{image_name}"


def drone_flight_plot_image_path(drone_mission_dir: str, flight_date: str, datetime: str, plot_id: str, image_name: str) -> str:
    """
    Generates the file path for a drone flight plot image based on the specified directory,
    date-time, plot identifier, and image name.

    Args:
        ortho_dir: The base directory where the plot images are stored.
        datetime: The datetime when the images were cropped.
        plot_id: A unique identifier for the plot.
        image_name: The name of the image file.

    Returns:
        The full file path for the specified drone flight plot image as a string.
    """
    plot_id = plot_id.lower()
    datetime = _format_datetime_string(datetime)
    image_name = image_name.lower()

    return f"{drone_mission_dir}/{flight_date}/plot_image/{datetime}/{plot_id}/f{image_name}"


def drone_plot_dem_path(drone_mission_dir: str, flight_date: str, datetime: str, plot_id: str, dem_name: str) -> str:
    """
    Generates a formatted file path for a drone flight's plot DEM (Digital Elevation Model).

    This function constructs a path to the DEM file of a particular plot for a given
    datetime, plot ID, and DEM name within a specified root directory for orthophotos.

    Parameters:
        ortho_dir: Root directory where orthophotos are stored.
        datetime: Date and time string in a certain format to identify the flight.
        plot_id: Identifier of the plot.
        dem_name: Name of the DEM file.

    Returns:
        A string representing the fully constructed path to the corresponding
        DEM file within the specified directory structure.
    """
    plot_id = plot_id.lower()
    datetime = _format_datetime_string(datetime)
    flight_date = _format_date_string(flight_date)
    dem_name = dem_name.lower()

    return f"{drone_mission_dir}/{flight_date}/plot_dem/{datetime}/{plot_id}/{dem_name}"


def drone_pipeline_outputs(drone_mission_dir: str, flight_date, ) -> str:
    """
    Generates a formatted file path for a drone flight's plot DEM (Digital Elevation Model).

    This function constructs a path to the DEM file of a particular plot for a given
    datetime, plot ID, and DEM name within a specified root directory for orthophotos.

    Parameters:
        ortho_dir: Root directory where orthophotos are stored.
        datetime: Date and time string in a certain format to identify the flight.
        plot_id: Identifier of the plot.
        dem_name: Name of the DEM file.

    Returns:
        A string representing the fully constructed path to the corresponding
        DEM file within the specified directory structure.
    """
    flight_date = _format_date_string(flight_date)
    return f"{drone_mission_dir}/{flight_date}/outputs"
