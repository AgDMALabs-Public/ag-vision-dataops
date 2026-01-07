import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
from databricks.sdk import WorkspaceClient
import os

from ag_vision.constants import paths
from ag_vision.data_io import aws_io, local_io, databricks_io
from ag_vision.core.image import AgImage
from ag_vision.core import img_qc as iq
from ag_vision.data_io import schemas as sch
import logging

logger = logging.getLogger(__name__)  # Use __name__ to get the module's name


class AgImageIngest:
    def __init__(self, platform: str = None, cloud_bucket: str = None, ingest_df: pd.DataFrame or None = None,
                 task: str = None):
        """

        """
        self.platform = platform
        self.cloud_bucket = cloud_bucket
        self.ingest_df = ingest_df
        self.task = task
        self.failed_upload = []

    def copy_data(self):
        """

        """
        if self.platform in ['local', 'db']:
            self.ingest_df['dir'] = self.ingest_df['new_path'].apply(lambda x: os.path.dirname(x))

            # Make sure that the new location exists to copy to
            unique_dirs = self.ingest_df['dir'].unique()
            print('making the new directories')
            for dir_name in tqdm(unique_dirs, total=len(unique_dirs)):
                os.makedirs(dir_name, exist_ok=True)

            # Copy the data to the new locations
            print('Copying the files')
            for idx, row in tqdm(self.ingest_df.iterrows(), total=len(self.ingest_df)):
                dist = shutil.copy(row['src_path'], row['dst_path'])
        else:
            logger.warning(f'The cloud platform need to be local or db')

    def upload_local_data_to_db(self, db_client: WorkspaceClient, chunk_size: int = 1024 * 1024, generate_metadata: bool = True):
        """

        """
        for idx, row in tqdm(self.ingest_df.iterrows(), total=len(self.ingest_df)):
            try:
                databricks_io.upload_file_with_progress(w=db_client,
                                                        local_file_path=row['src_path'],
                                                        volume_path=row['dst_path'],
                                                        chunk_size = chunk_size)

                if generate_metadata:
                    ag_img = AgImage(img_key=row['dst_path'],
                                     platform=self.platform)
                    ag_img.generate_metadata_key_from_img_key()
                    ag_img.initialize_metadata(device=row['device'],
                                               img_type=row['img_type'])
                    ag_img.add_image_id_to_metadata()
                    ag_img.metadata.path = row['dst_path']
                    ag_img.metadata.protocol_properties.protocol_name = row['protocol_name']
                    ag_img.metadata.protocol_properties.protocol_version = row['protocol_version']

                    ag_img.upload_metadata_to_cloud()

            except Exception as e:
                self.failed_upload.append(row['src_path'])
                logger.warning(f"{row['src_path']} Failed, Exception occurred while uploading data to DBFS: {e}")

    def add_season_column(self):
        """

        """
        for idx, row in tqdm(self.ingest_df.iterrows(), total=len(self.ingest_df)):
            self.ingest_df.loc[idx, 'season'] = paths.season_code(year=row['year'],
                                                                  crop=row['crop'],
                                                                  country=row['country'],
                                                                  time_of_year=str(row['time_of_year']))

    def validate_ingest_df(self):
        pass

