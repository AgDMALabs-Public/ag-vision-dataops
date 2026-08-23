import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_survey_table(file_list: list, project_index: int = 5) -> pd.DataFrame:
    """

    """
    file_list = list(file_list)

    survey_df = pd.DataFrame({'file_path': file_list})

    survey_df['project'] = [x.split('/')[project_index] for x in file_list]
    survey_df['site'] = [x.split('/')[project_index + 1] for x in file_list]
    survey_df['trial'] = [x.split('/')[project_index + 2] for x in file_list]
    survey_df['season'] = [x.split('/')[project_index + 3] for x in file_list]
    survey_df['field'] = [x.split('/')[project_index + 4] for x in file_list]
    survey_df['location'] = [x.split('/')[project_index + 5] for x in file_list]
    survey_df['protocol'] = [x.split('/')[project_index + 8] for x in file_list]
    survey_df['collection_date'] = [x.split('/')[project_index + 9] for x in file_list]
    survey_df['survey_id'] = [x.split('/')[-1].replace('.json', '') for x in file_list]

    return survey_df
