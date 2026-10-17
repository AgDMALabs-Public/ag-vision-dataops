import pandas as pd
import os

from typing import Iterator
from ag_vision.pipelines import image_processing as ip
from ag_vision.pipelines import annotation_processing as ap




def keep_records_with_missing_metadata_files(
    df: pd.DataFrame,
    path_col: str = "metadata_path"
    )-> pd.DataFrame:

    """
    Adds a boolean column 'metatada_file_exists' to df, then returns
    only the rows where the file does NOT already exist.
    """
    df = df.copy()
    df["metatada_file_exists"] = df[path_col].apply(os.path.exists)
    return df[df["metatada_file_exists"] == False]



def process_images_batch(
    iterator: Iterator[pd.DataFrame],
    sample_df: pd.DataFrame
    )-> Iterator[pd.DataFrame]:
    """
    Processes an iterator of pandas DataFrames (chunks) and yields processed results.
    """
    for pdf in iterator:
        # pdf is a pandas DataFrame chunk
        items = pdf["item"].tolist()

        if not items:
            # Yield an empty DataFrame with the correct columns if the chunk is empty
            yield pd.DataFrame(columns=sample_df.columns)
        else:
            # Process the items and yield the resulting DataFrame
            yield ip.generate_images_table(img_list=items,
                                           platform='db',
                                           project_index=6)
            

def process_annotations_batch(
    iterator: Iterator[pd.DataFrame],
    )-> Iterator[pd.DataFrame]:
    """
    Processes an iterator of pandas DataFrames (chunks) and yields processed results.
    """
    for pdf in iterator:
        # pdf is a pandas DataFrame chunk
        items = pdf["item"].tolist()

        if not items:
            # Yield an empty DataFrame with the correct columns if the chunk is empty
            yield ap.generate_annotations_table(anno_list=[])
        else:
            # Process the items and yield the resulting DataFrame
            yield ap.generate_annotations_table(anno_list=items)

