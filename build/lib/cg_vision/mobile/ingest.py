import pandas as pd
import logging
from uuid import uuid4
import os

from ag_vision.core.ingest import AgImageIngest
from ag_vision.constants import paths
from ag_vision.core import ingest_qc as iqc
from open_aglabs.core import constants as cst

INGEST_COLS = ['src_path', 'project_dir', 'site', 'trial', 'year', 'country', 'crop', 'time_of_year', 'season',
               'field', 'location', 'task', 'protocol', 'collection_date', 'id', 'plot_id', 'img_ext', 'event_type']

METADATA_COLS = ['image_type','sample_type', 'object_resolution', 'camera_make', 'camera_angle', 'camera_height_m',
                 'device', 'protocol_name', 'image_type', 'crop_type', 'growth_stage', 'soil_color', 'tillage_type',
                 'weed_pressure', 'notes']

class MobileImageIngest(AgImageIngest):
    def __init__(self, platform: str = None, cloud_bucket: str = None, ingest_df: pd.DataFrame or None = None,
                 event_type: str = None):
        # Call the initializer of the parent class 'Image' to handle the common attributes
        super().__init__(platform, cloud_bucket, ingest_df)

        self.event_type = event_type

    def generate_ingest_df(self, file_list):
        self.ingest_df = pd.DataFrame({'src_path': file_list})
        self.ingest_df['img_ext'] = self.ingest_df['src_path'].apply(lambda x: os.path.splitext(x)[1])

        for col in INGEST_COLS:
            if col not in self.ingest_df.columns:
                self.ingest_df[col] = None

    def generate_unique_image_ids(self):
        self.ingest_df['id'] = self.ingest_df['src_path'].apply(lambda x: str(uuid4()))

    def extract_img_id_from_src_path(self):
        self.ingest_df['id'] = self.ingest_df['src_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

    def validate_ingest_df(self):
        iqc.validate_df(df=self.ingest_df,
                        df_cols=INGEST_COLS)

        iqc.validate_column(df=self.ingest_df,
                            column_name='event_type',
                            approved_values=['scouting', 'trial'])

        iqc.validate_column(df=self.ingest_df,
                            column_name='year',
                            approved_values=cst.YEAR_LIST)

        iqc.validate_column(df=self.ingest_df,
                            column_name='country',
                            approved_values=cst.COUNTRY_CODES.keys())

        iqc.validate_column(df=self.ingest_df,
                            column_name='crop',
                            approved_values=cst.CROP_LIST)

        iqc.validate_column(df=self.ingest_df,
                            column_name='time_of_year',
                            approved_values=cst.TIME_OF_YEAR_LIST)


    def generate_dst_path_name(self):
        for idx, row in self.ingest_df.iterrows():
            loc_path = paths.location_path(project=row['project_dir'],
                                           site=row['site'],
                                           trial=row['trial'],
                                           season=row['season'],
                                           field=row['field'],
                                           location=row['location'])

            if row['event_type'] == 'scouting':
                self.ingest_df.loc[idx, 'dst_path'] = paths.scouting_image_path(location_path=loc_path,
                                                                                task=row['task'],
                                                                                protocol=row['protocol'],
                                                                                date=row['collection_date'],
                                                                                image_name=row['id'] + row['img_ext'])
            elif row['event_type'] == 'trial':
                self.ingest_df.loc[idx, 'dst_path'] = paths.trial_image_path(location_path=loc_path,
                                                                             task=row['task'],
                                                                             protocol=row['protocol'],
                                                                             date=row['collection_date'],
                                                                             plot_id=row['plot_id'],
                                                                             image_name=row['id'] +  + row['img_ext'])
            else:
                raise ValueError(f"The event type {row['event_type']} is not supported.")
