import numpy as np
import os

from open_aglabs.surveys.models import SurveyDataModel

from ag_vision.data_io import local_io, databricks_io
from ag_vision.constants import paths
import logging
from uuid import uuid4

logger = logging.getLogger(__name__)


class AgSurvey:
    def __init__(self, platform: str = None, cloud_bucket: str = None, location_key: str = None, survey_key: str = None,
                 survey: SurveyDataModel = None, audio_key: str = None, audio: np.ndarray = None,
                 audio_sr: float = None, text_key: str = None,
                 text: str = None, db_client=None):
        """
        Constructor for the SurveyDataModel class.

        Args:
            platform: The platform from which the survey was submitted.
            cloud_bucket: The name of the cloud bucket where the survey data is stored.
            survey_key: The key of the survey in the cloud bucket.
            audio: The audio recording of the survey participant.
            text: The transcribed text of the survey participant.
            survey: The SurveyDataModel object representing the survey.
        """
        self.cloud_client = db_client
        self.platform = platform
        self.cloud_bucket = cloud_bucket
        self.location_key = location_key
        self.survey_key = survey_key
        self.survey = survey
        self.audio_key = audio_key
        self.audio = audio
        self.audio_sr = audio_sr
        self.text_key = text_key
        self.text = text

    def __repr__(self):
        return f"AgSurvey(key={self.survey_key})"

    def generate_location_key(self, project, site, trial, season, field, location):
        self.location_key = paths.location_path(project=project,
                                                site=site,
                                                trial=trial,
                                                season=season,
                                                field=field,
                                                location=location)

    def generate_survey_key(self, task, protocol, date, f_name):
        """
        Generates a survey key based on provided task, protocol, date, and file name.

        Args:
            task: The task name.
            protocol: The protocol name.
            date: The date.
            f_name: The file name.

        Raises:
            AssertionError: If the location key is not set.

        Returns:
            None
        """
        assert self.location_key is not None, "location key needs to be set"

        self.survey_key = paths.survey_path(location_path=self.location_key,
                                            task=task,
                                            protocol=protocol,
                                            date=date,
                                            survey_obj_name=f_name)

    def load_audio_data(self):
        """
        Loads audio data based on the specified platform.

        Raises:
            AssertionError: If the audio key is not set.
        """
        assert self.audio_key is not None, "The audio key needs to be set."

        # Save to Databricks if platform is set to 'db'
        if self.platform == 'db':
            try:
                self.audio, self.audio_sr = databricks_io.read_audio_from_databricks(file_name=self.audio_key)

            except Exception as e:
                logger.warning(f'Exception occurred while loading audio from Databricks: {e}')

        # Save locally if platform is set to 'local'
        elif self.platform == 'local':
            try:
                self.audio, self.audio_sr = local_io.read_audio(file_path=self.audio_key)

            except Exception as e:
                logger.warning(f'Exception occurred while saving audio locally: {e}')

    def load_text_data(self, encoding: str = 'utf-8'):
        """
        Loads text data from Databricks or local storage.

        Args:
            encoding: The encoding of the text data. Defaults to 'utf-8'.

        Raises:
            AssertionError: If the text key is not set.

        Returns:
            The loaded text data.
        """
        assert self.text_key is not None, "The text key needs to be set."

        # Save to Databricks if platform is set to 'db'
        if self.platform == 'db':
            try:
                self.text = databricks_io.read_text_from_databricks(file_name=self.text_key,
                                                                    encoding=encoding)

            except Exception as e:
                logger.warning(f'Exception occurred while loading text from Databricks: {e}')

        # Save locally if platform is set to 'local'
        elif self.platform == 'local':
            try:
                self.text = local_io.read_text_data(file_path=self.text_key,
                                                    encoding=encoding)

            except Exception as e:
                logger.warning(f'Exception occurred while saving text locally: {e}')

    def save_audio(self, audio_key: str = None):
        """
        Saves the stored audio data (self.audio) locally and/or to Databricks.

        Args:
            audio_key: Optional key for the saved audio file. If None, self.audio_key is used.

        Returns:
            None
        """
        assert self.audio is not None, "Audio data (self.audio) must be loaded before saving."

        save_key = audio_key if audio_key is not None else self.audio_key
        if save_key is None:
            raise ValueError("Cannot determine the audio file key/path.")

        # Save to Databricks if platform is set to 'db'
        if self.platform == 'db':
            try:
                databricks_io.save_audio(audio_data=self.audio,
                                         file_path=save_key,
                                         sample_rate=self.audio_sr)

            except Exception as e:
                logger.warning(f'Exception occurred while saving audio to Databricks: {e}')

        # Save locally if platform is set to 'local'
        elif self.platform == 'local':
            try:
                local_io.save_audio(audio_data=self.audio,
                                    file_path=save_key,
                                    sample_rate=self.audio_sr)

            except Exception as e:
                logger.warning(f'Exception occurred while saving audio locally: {e}')

    def save_text(self):
        """
        Save the text associated with an audio object.

        **Summary:**
        This method saves the text associated with an audio object to Databricks or locally, depending on the platform setting.

        **Args:**
            self: The audio object.

        **Raises:**
            AssertionError: If the audio data or text key is not set.

        **Side Effects:**
        - Saves the text to Databricks or locally.
        - Logs any exceptions encountered during the saving process.
        """
        assert self.text is not None, "Audio data (self.audio) must be loaded before saving."
        assert self.text_key is not None, "The text key needs to be set"

        # Save to Databricks if platform is set to 'db'
        if self.platform == 'db':
            try:
                databricks_io.save_text_to_databricks(data=self.text,
                                                      file_name=self.text_key)

            except Exception as e:
                logger.warning(f'Exception occurred while saving text to Databricks: {e}')

        # Save locally if platform is set to 'local'
        elif self.platform == 'local':
            try:
                local_io.save_text_data(data=self.text,
                                        file_path=self.text_key)

            except Exception as e:
                logger.warning(f'Exception occurred while saving text locally: {e}')

    def load_survey_from_dict(self, survey_dict: dict):
        """
        Loads a survey from a dictionary.

        Args:
            survey_dict: A dictionary containing the survey data.

        Returns:
            None
        """
        try:
            self.survey = SurveyDataModel(**survey_dict)
        except Exception as e:
            self.survey = None
            logger.warning(f'Exception occurred while reading Survey: {e}')

    def read_survey(self):
        """
        Reads survey data from either DBFS or local storage.

        Raises:
            AssertionError: If `survey_key` is not set.

        Returns:
            None
        """
        assert self.survey_key is not None, "survey key needs to be set."

        if self.platform == 'db':
            if os.path.exists(self.survey_key):
                try:
                    m_data = databricks_io.read_json_from_databricks(file_name=self.survey_key)
                    self.survey = SurveyDataModel(**m_data)
                except Exception as e:
                    self.survey = None
                    logger.warning(f'Exception occurred while reading Survey from DBFS: {e}')

        elif self.platform == 'local':
            try:
                m_data = local_io.read_json(file_path=self.survey_key)
                self.survey = SurveyDataModel(**m_data)
            except Exception as e:
                self.survey = None
                logger.warning(f'Exception occurred while reading Survey from local: {e}')
        else:
            logger.warning(f'The cloud platform need to be local or db')

    def validate_survey(self):
        """
        Validates the survey data using Pydantic model.

        Args:
            self: The instance of the class.

        Returns:
            validated_metadata: The validated survey data as a SurveyDataModel object.

        Raises:
            RuntimeError: If the survey fails Pydantic validation check.
        """
        assert self.survey is not None, "Survey data is None"
        try:
            validated_metadata = SurveyDataModel(**self.survey.model_dump())
            logger.info("Survey successfully validated using Pydantic model.")
            return validated_metadata
        except Exception as e:
            raise RuntimeError(
                f"Survey failed Pydantic validation check during integrity verification: {type(e).__name__}: {e}")

    def save_survey(self):
        """
        Saves the survey data to the database or local storage.

        Args:
            self: The instance of the class.

        Raises:
            AssertionError: If the survey or survey key is not set.

        Returns:
            None
        """
        assert self.survey is not None, "Survey is none, will not save."
        assert self.survey_key is not None, "Survey key needs to be set."

        self.validate_survey()

        if self.platform == 'db':
            try:
                databricks_io.save_json_to_databricks(data=self.survey.model_dump(),
                                                      file_name=self.survey_key)

            except Exception as e:
                logger.warning(f'Exception occurred while saving metadata to DB: {e}')

        elif self.platform == 'local':
            try:
                local_io.save_json(data=self.survey.model_dump(),
                                   file_path=self.survey_key)
            except Exception as e:
                logger.warning(f'Exception occurred while saving metadata to local: {e}')

    def upload_survey_data_to_databricks(self, db_path: str):
        """
        Uploads survey data to Databricks.

        Args:
            db_path: The path to the Databricks file where the survey data will be uploaded.

        Raises:
            AssertionError: If the survey data is not set.

        """
        assert self.survey is not None, "There is no survey data to upload."

        self.validate_survey()
        try:
            databricks_io.upload_json_to_databricks(w=self.cloud_client,
                                                    data=self.survey.model_dump(),
                                                    file_name=db_path)
        except Exception as e:
            print(f"Failed to upload survey data to databricks {e}")

    def upload_survey_text_to_databricks(self, db_path: str):
        """
        Upload survey text to databricks.

        Args:
            db_path: The path to the survey text in databricks.

        Raises:
            AssertionError: If there is no text data to upload.

        Returns:
            None
        """
        assert self.text is not None, "there is no text data to upload."
        try:
            databricks_io.upload_text_to_databricks(w=self.cloud_client,
                                                    data=self.text,
                                                    file_name=db_path)
        except Exception as e:
            print(f"Failed to upload text data to databricks {e}")

    def upload_survey_audio_to_databricks(self, db_path: str):
        """
        Uploads survey audio to databricks.

        Args:
            db_path: Path to upload the audio data in databricks.

        Raises:
            AssertionError: If the audio key is not set.

        """
        assert self.audio_key is not None, 'The audio key needs to be set.'
        try:
            databricks_io.upload_wav_to_databricks(w=self.cloud_client,
                                                   local_file_path=self.audio_key,
                                                   dbfs_file_name=db_path)
        except Exception as e:
            print(f"Failed to uplaod Audio data to databricks {e}")

    def initialize_survey(self, collection_date: str):
        """
        Initializes a new survey object.

        Args:
            collection_date: The date of the survey collection.

        """
        assert self.survey_key is not None, "The survey key is not set"

        survey_dict = {
            "id": str(uuid4()),  # required
            "path": self.survey_key,
            "collection_date": collection_date,
            "trial_properties": {},
            "protocol_properties": {},
            "location_properties": {},
            "agronomic_properties": {},
            "answers": {},
            "followups": {},
            "notes": [{"message": "", 'author': ""}]
        }
        self.survey = SurveyDataModel(**survey_dict)
