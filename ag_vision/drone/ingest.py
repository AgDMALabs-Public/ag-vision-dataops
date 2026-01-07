import uuid

import pandas as pd
from tqdm import tqdm
import os

from ag_vision.core.ingest import AgImageIngest
from ag_vision.constants import paths
from ag_vision.data_io import local_io as lio
from ag_vision.data_io import databricks_io as dio
from open_aglabs.drone.model import DroneFlight

RAW_INGEST_COLS = ['src_path', 'file_type', 'camera']
PLOT_INGEST_COLS = ['src_path', 'plot_id', 'camera']


class DroneDataIngest(AgImageIngest):
    def __init__(self, platform: str, cloud_bucket: str, flight_date: str, plot_ingest_df: pd.DataFrame or None = None,
                 plot_boundary_key: str = None, gcp_key: str = None, orthomosaic_key: str = None, dem_key: str = None,
                 flight_metadata_key: str = None, flight_metadata: DroneFlight or None = None, cloud_client=None):
        """
        Represents a drone imagery flight operation, encapsulating details such as
        specific date, relevant metadata, and associated data files required for
        processing or uploading.

        Attributes:
            platform: the platform we are uploading data to (e.g. 'aws', 'db').
            cloud_bucket: The bucket name that the data will be uploaded to.
            flight_date: A string representing the date of the flight in the format
                YYYY-MM-DD.
            flight_metadata: An instance of DroneFlight or None containing detailed
                metadata for the flight.
            flight_metadata_key: the path to the local flight metadata file to upload.
            plot_boundary_key: The path to the local plot boundary file to upload, needs to be geojson.
            gcp_key: The path to the local ground control point file to upload, needs to be geojson.
            orthomosaic_key: The path to the local orthomosaic data to upload.
            dem_key: The path to the local DEM data to upload.
            cloud_client: The cloud client instance used for uploading files.
        """
        # Call the initializer of the parent class 'Image' to handle the common attributes
        super().__init__(platform, cloud_bucket)
        self.cloud_client = cloud_client
        self.drone_mission_dir = None
        self.flight_date = pd.to_datetime(flight_date).strftime("%Y-%m-%d")
        self.flight_metadata = flight_metadata
        self.flight_metadata_key = flight_metadata_key
        self.plot_boundary_key = plot_boundary_key
        self.gcp_key = gcp_key
        self.orthomosaic_key = orthomosaic_key
        self.dem_key = dem_key
        self.plot_ingest_df = plot_ingest_df
        self.raw_ingest_df = None

    def generate_raw_ingest_df(self, file_list):
        """
        Generates and initializes an ingestion DataFrame with metadata from the given list of file paths.
        The DataFrame includes essential columns such as source path, file name, and image extension,
        along with additional columns predefined in the INGEST_COLS list. It ensures all specified
        columns in INGEST_COLS are present in the DataFrame, initializing them with None if not already defined.

        Args:
            file_list (List[str]): List of file paths to generate the ingestion DataFrame.

        Returns:
            None
        """
        self.raw_ingest_df = pd.DataFrame({'src_path': file_list})
        self.raw_ingest_df['file_name'] = self.raw_ingest_df['src_path'].apply(lambda x: os.path.basename(x))
        self.raw_ingest_df['img_ext'] = self.raw_ingest_df['file_name'].apply(lambda x: os.path.splitext(x)[1])

        for col in RAW_INGEST_COLS:
            if col not in self.raw_ingest_df.columns:
                self.raw_ingest_df[col] = None

    def generate_plot_ingest_df(self, df: pd.DataFrame, generate_uuid: bool = True):
        """
        Generates and initializes an ingestion DataFrame with metadata from the given list of file paths.
        The DataFrame includes essential columns such as source path, file name, and image extension,
        along with additional columns predefined in the INGEST_COLS list. It ensures all specified
        columns in INGEST_COLS are present in the DataFrame, initializing them with None if not already defined.

        Args:
            file_list (List[str]): List of file paths to generate the ingestion DataFrame.
            generate_uuid (bool): Whether to generate a unique ID for each row in the DataFrame. This will be used as the new image name.
        Returns:
            None
        """
        self.plot_ingest_df = df.copy()

        for col in PLOT_INGEST_COLS:
            assert col in self.plot_ingest_df.columns, f"{col} is not in the raw ingest df, it needs {PLOT_INGEST_COLS}"

        self.plot_ingest_df['file_name'] = self.raw_ingest_df['src_path'].apply(lambda x: os.path.basename(x))
        self.plot_ingest_df['img_ext'] = self.raw_ingest_df['file_name'].apply(lambda x: os.path.splitext(x)[1])

        if generate_uuid:
            self.plot_ingest_df['new_file_name'] = self.plot_ingest_df['file_ext'].apply(lambda x: str(uuid.uuid4()) + '.' + x)
        else:
            self.plot_ingest_df['new_file_name'] = self.plot_ingest_df['file_name']

        self.plot_ingest_df['new_file_name'] = self.plot_ingest_df['new_file_name'].str.lower()

    def add_season_code_to_metadata(self, year: int, country: str, crop: str, time_of_year: str):
        """
        Adds a season code to the flight metadata's agronomic properties based on the provided inputs.

        Parameters:
            year (int): The year for which the season code should be computed.
            country (str): The country corresponding to the computation of the season code.
            crop (str): The crop type to include in the season code computation.
            time_of_year (str): The season or specific part of the year to consider.

        Returns:
            None
        """
        assert self.flight_metadata is not None, "Flight metadata cannot be None."
        self.flight_metadata.agronomic_properties.season_code = paths.season_code(year=year,
                                                                                  country=country,
                                                                                  crop=crop,
                                                                                  time_of_year=time_of_year)

    def generate_drone_mission_dir_path(self):
        """
        Generates the drone mission directory path based on flight metadata and cloud bucket details.
        """
        assert self.flight_metadata is not None, "flight_metadata cannot be None."
        assert self.cloud_bucket is not None, "cloud_bucket cannot be None."
        assert self.flight_metadata.location is not None, "flight_metadata.location cannot be None."
        assert self.flight_metadata.location.site is not None, "flight_metadata.location.site cannot be None."
        assert self.flight_metadata.trial_properties.name is not None, "flight_metadata.trial_properties.name cannot be None."
        assert self.flight_metadata.agronomic_properties.season_code is not None, "flight_metadata.agronomic_properties.season_code cannot be None."
        assert self.flight_metadata.location.field is not None, "flight_metadata.location.field cannot be None."
        assert self.flight_metadata.location.location is not None, "flight_metadata.location.location cannot be None."
        assert self.flight_metadata.name is not None, "flight_metadata.name cannot be None."

        loc_path = paths.location_path(project=self.cloud_bucket,
                                       site=self.flight_metadata.location.site,
                                       trial=self.flight_metadata.trial_properties.name,
                                       season=self.flight_metadata.agronomic_properties.season_code,
                                       field=self.flight_metadata.location.field,
                                       location=self.flight_metadata.location.location)

        self.drone_mission_dir = paths.drone_mission_dir(location_path=loc_path,
                                                         mission_name=self.flight_metadata.name)

    def load_metadata_from_dict(self, metadata_dict: dict):
        """
        Loads metadata from the provided dictionary and initializes the flight metadata Pydantic model.
        """
        self.flight_metadata = DroneFlight(**metadata_dict)

    def generate_raw_image_dst_path_name(self):
        """
        Generates and assigns the destination path for raw image or flight data files.

        Raises:
            ValueError: If the `file_type` of any row in `ingest_df` is not 'raw_image' or
                'flight_data'.
        """
        assert self.drone_mission_dir is not None, "drone_mission_dir cannot be None."
        assert self.flight_date is not None, "flight_date cannot be None."

        for idx, row in self.raw_ingest_df.iterrows():
            self.raw_ingest_df.loc[idx, 'dst_path'] = paths.drone_raw_flight_data(
                drone_mission_dir=self.drone_mission_dir,
                flight_date=self.flight_date,
                camera=row['camera'],
                file_name=str(row['file_name']))

    def generate_plot_image_dst_path_name(self):
        """
        Generates and assigns the destination path for raw image or flight data files.

        Raises:
            ValueError: If the `file_type` of any row in `ingest_df` is not 'raw_image' or
                'flight_data'.
        """
        assert self.drone_mission_dir is not None, "drone_mission_dir cannot be None."
        assert self.flight_date is not None, "flight_date cannot be None."

        for idx, row in self.plot_ingest_df.iterrows():
            self.plot_ingest_df.loc[idx, 'dst_path'] = paths.drone_flight_plot_image_path(
                drone_mission_dir=self.drone_mission_dir,
                flight_date=self.flight_date,
                camera=row['camera'],
                datetime=row['file_generation_datetime'],
                plot_id=row['plot_id'],
                file_name= str(row['file_name']))

    def save_flight_metadata_to_json_local(self):
        """
        Saves flight metadata to a local JSON file.

        Raises:
            FileNotFoundError: If the file path specified is invalid.
            IOError: If an error occurs while writing to the file.
        """
        assert self.flight_metadata is not None, "Flight metadata cannot be None."
        assert self.flight_metadata_key is not None, "Flight metadata key cannot be None."

        lio.save_json(data=self.flight_metadata.model_dump(),
                      file_path=self.flight_metadata_key)

    def upload_metadata_to_db(self):
        """
        Uploads flight metadata to the database.

        Raises:
            AssertionError: If any of the required metadata attributes
                (`flight_metadata`, `drone_mission_dir`, `flight_metadata_key`) are None.
        """
        assert self.flight_metadata is not None, "flight_metadata cannot be None."
        assert self.drone_mission_dir is not None, "drone_mission_dir cannot be None."
        assert self.flight_metadata_key is not None, "flight_metadata_key cannot be None."

        upload_path = paths.drone_flight_details_path(self.drone_mission_dir,
                                                      flight_date=self.flight_date)

        dio.upload_file_with_progress(w=self.cloud_client,
                                      local_file_path=self.flight_metadata_key,
                                      volume_path=upload_path)

    def upload_plot_boundary_to_db(self):
        """
        Uploads a plot boundary file to a specified database using a cloud client.

        The method ensures that required attributes are set and validates the file format
        before uploading it to the specified path in the cloud storage. The file is uploaded
        with progress tracking.

        Raises:
        AssertionError: If the plot boundary key or drone mission directory is not set,
        or if the plot boundary key is not a '.geojson' file.
        """
        assert self.plot_boundary_key is not None, f"plot boundary key cannot be None."
        assert self.drone_mission_dir is not None, f"drone mission path cannot be None."
        assert '.geojson' in self.plot_boundary_key, f"plot boundary key must be a geojson file."

        upload_path = paths.drone_plot_boundary_path(drone_mission_dir=self.drone_mission_dir)
        print(f"Saving to {upload_path}")

        dio.upload_file_with_progress(w=self.cloud_client,
                                      local_file_path=self.plot_boundary_key,
                                      volume_path=upload_path)

    def upload_gcp_to_db(self):
        """
        Uploads a ground control point file to a specific location in a database using a cloud client.

        This method ensures that necessary parameters related to plot boundary and drone mission
        are provided and valid. It uploads the file to a computed path using a cloud client's
        upload functionality with progress tracking.

        Raises:
        AssertionError: If `plot_boundary_key` is None.
        AssertionError: If `drone_mission_dir` is None.
        AssertionError: If `plot_boundary_key` does not contain '.geojson'.
        """
        assert self.plot_boundary_key is not None, f"plot boundary key cannot be None."
        assert self.drone_mission_dir is not None, f"drone mission path cannot be None."
        assert '.geojson' in self.plot_boundary_key, f"gcp key must be a txt file."

        upload_path = paths.drone_mission_ground_control_point_path(drone_mission_dir=self.drone_mission_dir)
        print(f"Saving to {upload_path}")

        dio.upload_file_with_progress(w=self.cloud_client,
                                      local_file_path=self.gcp_key,
                                      volume_path=upload_path)

    def upload_raw_flight_data_to_db(self):
        """
        Uploads raw flight data to a database, iterating through a DataFrame
        and updating the status of each file upload.

        Attributes:
            ingest_df (DataFrame): A pandas DataFrame containing the details of files
            to be uploaded. Each row should include 'src_path' and 'dst_path' columns
            specifying the source path and destination path, respectively.

        Raises:
            Exception: Captures and logs any exceptions occurring during file
            uploads. The error details are recorded in the 'status' column of the
            DataFrame.
        """
        for idx, row in tqdm(self.raw_ingest_df.iterrows()):
            try:
                dio.upload_file_with_progress(w=self.cloud_client,
                                              local_file_path=row['src_path'],
                                              volume_path=row['dst_path'])
                self.raw_ingest_df.loc[idx, 'status'] = "success"
            except Exception as e:
                print(f'Failed to upload file {row["src_path"]} --TO--: {row["dst_path"]}')
                print(f'The error is: {e}')
                self.raw_ingest_df.loc[idx, 'status'] = str(e)

        def upload_plot_images_to_db(self):
            """
            Uploads raw flight data to a database, iterating through a DataFrame
            and updating the status of each file upload.

            Attributes:
                ingest_df (DataFrame): A pandas DataFrame containing the details of files
                to be uploaded. Each row should include 'src_path' and 'dst_path' columns
                specifying the source path and destination path, respectively.

            Raises:
                Exception: Captures and logs any exceptions occurring during file
                uploads. The error details are recorded in the 'status' column of the
                DataFrame.
            """
            for idx, row in tqdm(self.plot_ingest_df.iterrows()):
                try:
                    dio.upload_file_with_progress(w=self.cloud_client,
                                                  local_file_path=row['src_path'],
                                                  volume_path=row['dst_path'])
                    self.plot_ingest_df.loc[idx, 'status'] = "success"
                except Exception as e:
                    print(f'Failed to upload file {row["src_path"]} --TO--: {row["dst_path"]}')
                    print(f'The error is: {e}')
                    self.plot_ingest_df.loc[idx, 'status'] = str(e)

    def upload_orthomosaic_to_db(self, method: str, ortho_date: str, camera: str, file_name: str):
        """
        Uploads an orthomosaic image to the database using the specified parameters. Ensures that
        the necessary keys and paths are not None before proceeding with the upload. The upload
        path is dynamically constructed based on the provided method, orthomosaic date, and image
        name.

        Parameters:
            method: str
                The processing method used for orthomosaic generation Eg: Agisoft.
            ortho_date: str
                The date the orthomosaic image was generatedin string format.
            file_name: str
                file_name: The name used to save the file in the cloud.
            camera: str
                camera: The camera used for the orthomosaic image eg: rgn. multi-spec, thermal.

        Raises:
            AssertionError: If orthomosaic_key or drone_mission_dir is None.
        """
        assert self.orthomosaic_key is not None, f"Ortho key cannot be None."
        assert self.drone_mission_dir is not None, f"drone mission path cannot be None."

        upload_path = paths.drone_flight_orthomosaic_path(drone_mission_dir=self.drone_mission_dir,
                                                          flight_date=self.flight_date,
                                                          method=method,
                                                          ortho_date=ortho_date,
                                                          camera=camera,
                                                          image_name=file_name)
        print(f"Saving to {upload_path}")

        dio.upload_file_with_progress(w=self.cloud_client,
                                      local_file_path=self.orthomosaic_key,
                                      volume_path=upload_path)

    def upload_dem_to_db(self, method: str, dem_date: str, file_name: str):
        """
        Uploads a Digital Elevation Model (DEM) to the database. This function uses a provided
        cloud client to upload a DEM file associated with a drone mission. The DEM's associated
        date, method, and image name are used to determine the save path in the database.

        Args:
            method: The method name as a string used for demarcating the DEM file path.
            dem_date: A string representing the DEM's generatopm date.
            file_name: The name used to save the file in the cloud.

        Raises:
            AssertionError: If `self.dem_key` is None.
            AssertionError: If `self.drone_mission_dir` is None.
        """
        assert self.dem_key is not None, f"dem key cannot be None."
        assert self.drone_mission_dir is not None, f"drone mission path cannot be None."

        upload_path = paths.drone_flight_dem_path(drone_mission_dir=self.drone_mission_dir,
                                                  flight_date=self.flight_date,
                                                  method=method,
                                                  dem_date=dem_date,
                                                  image_name=file_name)
        print(f"Saving to {upload_path}")

        dio.upload_file_with_progress(w=self.cloud_client,
                                      local_file_path=self.dem_key,
                                      volume_path=upload_path)
